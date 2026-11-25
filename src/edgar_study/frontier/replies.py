"""Decode and validate raw model replies."""

import json
import math

from edgar_study.frontier.grammar import InvalidEquation


class InvalidReply(InvalidEquation):
    pass


def decode_json(reply):
    if not isinstance(reply, str) or not reply.strip():
        raise InvalidReply("empty_reply", "No proposal text returned.")
    text = reply.strip()
    if text.startswith("```") and text.endswith("```") and "\n" in text:
        text = text.split("\n", 1)[1][:-3]

    def reject_number(value):
        raise InvalidReply("nonfinite_reply_number", "Reply contains a nonfinite JSON number.")

    try:
        return json.loads(text, parse_constant=reject_number, parse_float=_finite_float)
    except InvalidReply:
        raise
    except (ValueError, RecursionError) as exc:
        raise InvalidReply("invalid_reply_json", "Reply is not readable JSON.") from exc


def equation_list(values, k, *, exact=True):
    if not isinstance(values, list) or len(values) > k or (exact and len(values) != k):
        count = f"exactly {k}" if exact else f"at most {k}"
        raise InvalidReply("invalid_reply_count", f"Expected {count} proposals.")
    return values


def decode(reply, k):
    return equation_list(decode_json(reply), k)


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise InvalidReply("nonfinite_reply_number", "Reply contains a nonfinite JSON number.")
    return number
