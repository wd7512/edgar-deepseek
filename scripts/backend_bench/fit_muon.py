"""Muon (momentum orthogonalised by Newton-Schulz) in PyTorch, CPU or Apple GPU.

Muon is designed for the 2-D weight matrices of a network. These fits have 5 or 11 parameters per
cell, so the update is either orthogonalised per cell (a 1 x K matrix, which reduces to normalised
momentum) or over the whole (cells x K) parameter block at once.
"""

import argparse
import math

import torch
from common import load, summarise, time_call
from fit_torch import batched, make_model, synchronize


def zeropower(matrix, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    rows, cols = matrix.shape[-2], matrix.shape[-1]
    x = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = rows > cols
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        update = b * gram + c * (gram @ gram)
        x = a * x + update @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x


def fit_muon(program, x, y, p0, steps, learning_rate, momentum, mode, device):
    single = make_model(program)
    p = p0.clone().requires_grad_(True)
    buffer = torch.zeros_like(p)
    best = p0.clone()
    best_loss = math.inf
    done = 0
    for _ in range(steps):
        loss = ((batched(single, x, p) - y) ** 2).mean()
        (gradient,) = torch.autograd.grad(loss, p)
        buffer = buffer.lerp(gradient, 1 - momentum)
        direction = gradient.lerp(buffer, momentum)
        if mode == "per-cell":
            ortho = zeropower(direction[:, None, :]).squeeze(1)
            scale = max(1.0, 1.0 / direction.shape[1]) ** 0.5
        else:
            ortho = zeropower(direction[None, :, :]).squeeze(0)
            scale = max(1.0, direction.shape[1] / direction.shape[0]) ** 0.5
        with torch.no_grad():
            p -= learning_rate * scale * ortho
        value = float(loss.detach())
        if not math.isfinite(value) or not bool(torch.isfinite(p).all()):
            break
        done += 1
        if value < best_loss:
            best_loss = value
            best = p.detach().clone()
    synchronize(device)
    return best, done


def subset(program, cells, device):
    data = load(program)
    return (
        data,
        torch.as_tensor(data["x_tr"][:cells], device=device),
        torch.as_tensor(data["y_tr"][:cells], device=device),
        torch.as_tensor(data["p0"][:cells], device=device),
    )


def choose_rate(program, device, cells, steps, momentum, mode, rates):
    data, x, y, p0 = subset(program, cells, device)
    best = None
    for rate in rates:
        params, _ = fit_muon(program, x, y, p0, steps, rate, momentum, mode, device)
        loss = float(((batched(make_model(program), x, params) - y) ** 2).mean())
        if math.isfinite(loss) and (best is None or loss < best[0]):
            best = (loss, rate)
    return best[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--programs", nargs="+", default=["double-gaussian", "winner"])
    parser.add_argument("--modes", nargs="+", default=["per-cell", "global"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tune-cells", type=int, default=100)
    parser.add_argument("--tune-steps", type=int, default=200)
    parser.add_argument("--rates", nargs="+", type=float, default=[3e-3, 1e-2, 3e-2])
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    for name in args.programs:
        for mode in args.modes:
            rate = choose_rate(
                name, device, args.tune_cells, args.tune_steps, args.momentum, mode, args.rates
            )
            data = load(name)
            x = torch.as_tensor(data["x_tr"], device=device)
            y = torch.as_tensor(data["y_tr"], device=device)
            p0 = torch.as_tensor(data["p0"], device=device)

            def call(x=x, y=y, p0=p0, rate=rate, mode=mode, name=name):
                return fit_muon(name, x, y, p0, args.steps, rate, args.momentum, mode, device)

            seconds, (params, iterations) = time_call(call, args.repeats)
            summarise(
                data,
                params.cpu().numpy(),
                seconds,
                iterations,
                "torch-muon",
                args.device,
                "muon-" + mode,
                {"converged": 0, "learning_rate": rate, "torch": torch.__version__},
            )


if __name__ == "__main__":
    main()
