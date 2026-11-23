"""Levenberg-Marquardt and Adam in PyTorch, on the CPU or the Apple GPU (MPS)."""

import argparse
import math

import torch
from common import (
    LM_MAX_ITERATIONS,
    LM_MAX_STEP,
    LM_TOLERANCE,
    MAX_DAMPING,
    load,
    summarise,
    time_call,
)
from torch.func import jacfwd, vmap


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def peak(x, phi, amplitude, sigma_plus, sigma_minus, exponent):
    d = torch.atan2(torch.sin(x - phi), torch.cos(x - phi))
    sigma = torch.where(
        d >= 0,
        torch.clamp(sigma_plus, 0.05, 3.0),
        torch.clamp(sigma_minus, 0.05, 3.0),
    )
    p = torch.clamp(exponent, 0.5, 8.0) if torch.is_tensor(exponent) else min(max(exponent, 0.5), 8.0)
    u = torch.abs(d) / sigma
    return torch.clamp(amplitude, min=0) * torch.exp(-0.5 * u**p)


def make_model(program):
    winner = program == "winner"

    def single(x, p):
        if winner:
            phi1, phi2, b = p[0], p[1], p[2]
            a1, a2 = p[3], p[4]
            s1p, s1m, s2p, s2m = p[5], p[6], p[7], p[8]
            p1, p2 = p[9], p[10]
            return torch.clamp(b, min=0) + peak(x, phi1, a1, s1p, s1m, p1) + peak(x, phi2, a2, s2p, s2m, p2)
        phi, b, a1, a2, s = p[0], p[1], p[2], p[3], p[4]
        return (
            torch.clamp(b, min=0)
            + peak(x, phi, a1, s, s, 2.0)
            + peak(
                x,
                phi + torch.full_like(phi, 3.141592653589793),
                a2,
                s,
                s,
                2.0,
            )
        )

    return single


def batched(single, x, p):
    return vmap(single, in_dims=(0, 0))(x, p)


def batched_jacobian(single, x, p):
    def one(x_row, p_row):
        return jacfwd(lambda q: single(x_row, q))(p_row)

    return vmap(one, in_dims=(0, 0))(x, p)


def peak_parts(x, phi, amplitude, sigma_plus, sigma_minus, exponent):
    """Value and analytic partials of one peak, broadcasting (cells, 1) params over (cells, trials)."""
    d = torch.atan2(torch.sin(x - phi), torch.cos(x - phi))
    sigma = torch.where(
        d >= 0,
        torch.clamp(sigma_plus, 0.05, 3.0),
        torch.clamp(sigma_minus, 0.05, 3.0),
    )
    p = torch.clamp(exponent, 0.5, 8.0) if torch.is_tensor(exponent) else min(max(exponent, 0.5), 8.0)
    absolute = torch.abs(d)
    u = absolute / sigma
    q = u**p
    e = torch.exp(-0.5 * q)
    gain = torch.clamp(amplitude, min=0)
    value = gain * e
    active = (amplitude > 0).to(e.dtype)
    dphi = 0.5 * gain * e * p * torch.sign(d) * q / (absolute + 1e-12)
    dsigma = 0.5 * gain * e * q * p / sigma
    ds_plus = torch.where(d >= 0, dsigma, torch.zeros_like(dsigma))
    ds_minus = torch.where(d < 0, dsigma, torch.zeros_like(dsigma))
    dp = -0.5 * gain * e * q * torch.log(u + 1e-12)
    return value, dphi, e * active, ds_plus, ds_minus, dp


def analytic_jacobian(program, x, p):
    cells, trials = x.shape
    K = p.shape[1]
    out = torch.zeros((cells, trials, K), dtype=p.dtype, device=p.device)
    if program == "winner":
        first = peak_parts(x, p[:, 0:1], p[:, 3:4], p[:, 5:6], p[:, 6:7], p[:, 9:10])
        second = peak_parts(x, p[:, 1:2], p[:, 4:5], p[:, 7:8], p[:, 8:9], p[:, 10:11])
        out[..., 0] = first[1]
        out[..., 1] = second[1]
        out[..., 2] = (p[:, 2:3] > 0).to(p.dtype)
        out[..., 3] = first[2]
        out[..., 4] = second[2]
        out[..., 5] = first[3]
        out[..., 6] = first[4]
        out[..., 7] = second[3]
        out[..., 8] = second[4]
        out[..., 9] = first[5]
        out[..., 10] = second[5]
        return out
    shared = torch.full_like(p[:, 4:5], 3.141592653589793)
    first = peak_parts(x, p[:, 0:1], p[:, 2:3], p[:, 4:5], p[:, 4:5], 2.0)
    second = peak_parts(x, p[:, 0:1] + shared, p[:, 3:4], p[:, 4:5], p[:, 4:5], 2.0)
    out[..., 0] = first[1] + second[1]
    out[..., 1] = (p[:, 1:2] > 0).to(p.dtype)
    out[..., 2] = first[2]
    out[..., 3] = second[2]
    out[..., 4] = first[3] + first[4] + second[3] + second[4]
    return out


def fit_lm(single, x, y, p0, bounded, max_iterations, device, jacobian_mode="autodiff", program=None):
    p = p0.clone()
    damping = torch.full((p.shape[0],), 1e-3, device=device)
    cost = ((batched(single, x, p) - y) ** 2).sum(dim=1)
    done = torch.zeros(p.shape[0], dtype=torch.bool, device=device)
    iterations = 0
    for iteration in range(max_iterations):
        if bounded and bool(done.all()):
            break
        residuals = batched(single, x, p) - y
        if jacobian_mode == "analytic":
            jac = analytic_jacobian(program, x, p)
        else:
            jac = batched_jacobian(single, x, p)
        gradient = torch.einsum("ctk,ct->ck", jac, residuals)
        hessian = torch.einsum("ctk,ctl->ckl", jac, jac)
        diagonal = torch.diagonal(hessian, dim1=1, dim2=2)
        scale = torch.maximum(diagonal, 1e-9 * diagonal.max(dim=1, keepdim=True).values + 1e-12)
        eye = torch.eye(p.shape[1], device=device)
        damped = hessian + (damping[:, None] * scale)[:, :, None] * eye
        step = torch.linalg.solve(damped, -gradient.unsqueeze(-1)).squeeze(-1)
        step = torch.where(torch.isfinite(step), step, torch.zeros_like(step))
        capped = torch.zeros(p.shape[0], dtype=torch.bool, device=device)
        if bounded:
            biggest = step.abs().max(dim=1, keepdim=True).values
            capped = biggest[:, 0] > LM_MAX_STEP
            step = step * torch.minimum(
                torch.ones_like(biggest), LM_MAX_STEP / torch.clamp(biggest, min=1e-30)
            )
        candidate = p + step
        new_cost = ((batched(single, x, candidate) - y) ** 2).sum(dim=1)
        accept = ~done & (new_cost < cost)
        if bounded:
            converged = (accept & (cost - new_cost <= LM_TOLERANCE * cost)) | (damping >= MAX_DAMPING)
            raised = torch.where(capped, damping * 2, damping / 3)
            update = torch.clamp(torch.where(accept, raised, damping * 4), 1e-7, MAX_DAMPING)
            damping = torch.where(done, damping, update)
            done = done | converged
        else:
            damping = torch.clamp(torch.where(accept, damping / 3, damping * 4), 1e-7, MAX_DAMPING)
        p = torch.where(accept[:, None], candidate, p)
        cost = torch.where(accept, new_cost, cost)
        iterations = iteration + 1
    synchronize(device)
    return p, iterations, int(done.sum())


def fit_adam(single, x, y, p0, steps, learning_rate=3e-3):
    p = p0.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([p], lr=learning_rate)
    best = p0.clone()
    best_loss = float("inf")
    steps_done = 0
    for _ in range(steps):
        optimiser.zero_grad()
        loss = ((batched(single, x, p) - y) ** 2).mean()
        loss.backward()
        optimiser.step()
        value = float(loss.detach())
        if not math.isfinite(value) or not bool(torch.isfinite(p).all()):
            break
        steps_done += 1
        if value < best_loss:
            best_loss = value
            best = p.detach().clone()
    synchronize(p.device)
    return best, steps_done, 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--programs", nargs="+", default=["double-gaussian", "winner"])
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--algorithms", nargs="+", default=["lm", "lm-bounded", "adam"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=LM_MAX_ITERATIONS)
    parser.add_argument("--adam-steps", type=int, default=1000)
    parser.add_argument("--jacobian", default="autodiff", choices=["autodiff", "analytic"])
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    for name in args.programs:
        data = load(name)
        single = make_model(name)
        x_all = torch.as_tensor(data["x_tr"], device=device)
        y_all = torch.as_tensor(data["y_tr"], device=device)
        p0 = torch.as_tensor(data["p0"], device=device)
        for algorithm in args.algorithms:
            if algorithm == "adam":
                seconds, (params, iterations, converged) = time_call(
                    lambda single=single, x_all=x_all, y_all=y_all, p0=p0: fit_adam(
                        single, x_all, y_all, p0, args.adam_steps
                    ),
                    args.repeats,
                )
            else:
                bounded = algorithm == "lm-bounded"
                seconds, (params, iterations, converged) = time_call(
                    lambda bounded=bounded, single=single, x_all=x_all, y_all=y_all, p0=p0, name=name: fit_lm(
                        single,
                        x_all,
                        y_all,
                        p0,
                        bounded,
                        args.iterations,
                        device,
                        args.jacobian,
                        name,
                    ),
                    args.repeats,
                )
            summarise(
                data,
                params.detach().cpu().numpy(),
                seconds,
                iterations,
                "torch",
                args.device,
                algorithm,
                {"converged": converged, "torch": torch.__version__, "jacobian": args.jacobian},
            )


if __name__ == "__main__":
    main()
