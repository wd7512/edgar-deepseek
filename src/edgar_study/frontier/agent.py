"""A small agentic loop for one frontier batch: the model proposes, may check, and repairs failures.

The model sees the batch prompt and images once. Each later turn is a short message in the same
conversation:
- a validity report, when the model asks to check equations;
- the reason each equation could not be used, with a request to replace it;
- a request to resend a reply that could not be read.

Each follow-up costs one turn of `patience`. When patience runs out, the batch keeps what it has.
Feedback says whether an equation can be used, never how well anything fits.
"""

import json
from collections.abc import Iterable, Sequence
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from edgar_study.frontier.grammar import InvalidEquation, compile_equation
from edgar_study.frontier.replies import InvalidReply, decode_json
from edgar_study.frontier.warm import complete_map

PATIENCE = 3
LARGEST_START_VALUE = 1e6
FRESH = "fresh"


class Session(Protocol):
    """What the loop needs from the run: reading replies, using equations and checking them."""

    def decode(self, reply: str, expected: int | str | None) -> list:
        """Proposals from `reply`, in the format `expected` names; raises `InvalidReply` if unreadable."""
        ...

    def submit(self, proposal: dict, replaces: int | None, turn: int) -> tuple[str, str] | None:
        """Use the proposal from follow-up `turn` (0 for the first reply); return None if it was fitted,
        else (reason code, detail)."""
        ...

    def check(self, proposals: list) -> str:
        """A validity report for proposals the model wants checked before submitting."""
        ...


def check_request(reply: str | None):
    """The equations in a `{"check": [...]}` reply, or None if the reply is not a check request."""
    try:
        value = decode_json(reply)
    except InvalidReply:
        return None
    if isinstance(value, dict) and set(value) == {"check"}:
        return value["check"] if isinstance(value["check"], list) else []
    return None


def _start_problems(equation):
    grid = jnp.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    start = jnp.asarray(list(equation.initial.values()), dtype=jnp.float32)

    def curve(params):
        return jnp.broadcast_to(equation.model(grid, *params), grid.shape)

    values = np.asarray(curve(start))
    if not np.isfinite(values).all():
        return ["nonfinite_start — values are undefined somewhere in 0..2*pi at the starting values."], []
    if np.max(np.abs(values)) > LARGEST_START_VALUE:
        return [
            f"overflow_start — values exceed {LARGEST_START_VALUE:g} at the starting values; "
            "fitting such an equation overflows."
        ], []
    gradient = np.asarray(jax.jacfwd(curve)(start))
    if not np.isfinite(gradient).all():
        return [], [
            "The slope with respect to some parameter is undefined at the starting values "
            "(for example abs or sqrt at zero); move the start away from that point."
        ]
    return [], []


def check_proposal(proposal, *, texts: Iterable[str] = (), keys: Iterable[str] = (), parents: Sequence = ()):
    """Score-free validity of one proposal: problems that stop it being used, and notes about it."""
    try:
        equation = compile_equation(proposal)
    except InvalidEquation as exc:
        return {"problems": [f"{exc.code} — {exc}"], "notes": []}
    problems, notes = [], []
    if equation.text.strip() in set(texts):
        problems.append("exact_repeat — this equation was already proposed.")
    elif equation.key in set(keys):
        problems.append(
            "duplicate — the same structure as an earlier equation, up to renaming or reordering."
        )
    found, warnings = _start_problems(equation)
    problems += found
    notes += warnings
    for parent in parents:
        nesting = complete_map(equation, parent)
        if nesting is None:
            notes.append(
                f"It does not contain {parent.node_id} as a special case: no setting of its parameters "
                f"reproduces {parent.node_id}."
            )
        elif nesting.declared:
            notes.append(f"Your nesting map for {parent.node_id} reproduces it.")
        else:
            notes.append(
                f"Your nesting map for {parent.node_id} is missing or does not reproduce it; this one does: "
                f"{json.dumps(nesting.bindings)}."
            )
    return {"problems": problems, "notes": notes}


def check_report(results: Sequence[dict]) -> str:
    """Format check results, one line per equation."""
    if not results:
        return 'Check results: nothing to check. Send {"check": [equation objects]}.'
    lines = ["Check results:"]
    for position, result in enumerate(results, start=1):
        status = "cannot be used: " + "; ".join(result["problems"]) if result["problems"] else "usable."
        notes = (" " + " ".join(result["notes"])) if result["notes"] else ""
        lines.append(f"- Equation {position}: {status}{notes}")
    return "\n".join(lines)


def _wanted(expected, k):
    if expected is None:
        return "your answer in the format the first message asked for"
    if expected == FRESH:
        return f"a JSON array of up to {k} equation objects"
    return f"a JSON array of exactly {expected} replacement equation objects, in the listed order"


def _instruction(text, expected, k, last):
    tail = f"Reply with {_wanted(expected, k)}"
    tail += (
        "; this is your last turn, so checking is no longer available."
        if last
        else (', or check equations first with {"check": [...]}.')
    )
    return f"{text}\n{tail}"


async def converse(proposer, batch, prompt, session: Session, *, k: int, patience: int = PATIENCE) -> dict:
    """Run one batch's conversation. Returns the first reply, each follow-up turn and how it ended."""
    reply = await proposer.submit(batch, prompt)
    transcript = {"reply": reply, "turns": []}
    history: list[tuple[str, str]] = []
    expected: int | str | None = None
    labels: list[int] | None = None
    fitted = 0
    while True:
        requested = check_request(reply)
        if requested is not None:
            if len(history) >= patience:
                transcript["ended"] = "check_without_answer"
                return transcript
            kind, text = "check", session.check(requested)
        else:
            try:
                proposals = session.decode(reply, expected)
            except InvalidReply as exc:
                kind, text = "unreadable", f"{exc.code}: {exc} Your reply could not be read."
            else:
                positions = labels or list(range(1, len(proposals) + 1))
                failures = []
                for position, proposal in zip(positions, proposals, strict=False):
                    replaces = position if isinstance(expected, int) else None
                    failure = session.submit(proposal, replaces, len(history))
                    if failure is None:
                        fitted += 1
                    else:
                        failures.append((position, *failure))
                if failures:
                    kind = "replace"
                    text = "These equations could not be used:\n" + "\n".join(
                        f"- Proposal {position}: {code} — {detail}" for position, code, detail in failures
                    )
                    text += "\nGive one different equation for each, keeping the same idea."
                    expected, labels = len(failures), [position for position, _, _ in failures]
                elif fitted == 0:
                    kind = "fresh"
                    text = (
                        "No equation from your reply could be used. Give materially different equations; "
                        "do not repeat earlier ones."
                    )
                    expected, labels = FRESH, None
                else:
                    transcript["ended"] = "done"
                    return transcript
        if len(history) >= patience:
            transcript["ended"] = f"out_of_patience_after_{kind}"
            return transcript
        instruction = _instruction(text, expected, k, last=len(history) + 1 >= patience)
        history.append((reply or "", instruction))
        reply = await proposer.retry(batch, prompt, list(history))
        transcript["turns"].append({"kind": kind, "instruction": instruction, "reply": reply})
