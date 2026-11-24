"""Find a learned program's two peak exponents and the range the program itself clips them to."""

import re

import numpy as np

PAIRS = (
    ("exponent_1", "exponent_2"),
    ("shape_power_1", "shape_power_2"),
    ("power_1", "power_2"),
    ("shape_1", "shape_2"),
)


def exponent_pair(names):
    """The names of the two peak exponents among a program's parameter names, or None."""
    return next((pair for pair in PAIRS if pair[0] in names and pair[1] in names), None)


def clip_bounds(code, name):
    """The `(low, high)` a program passes to `clip(name, low, high)`, or `clamp(low, name, high)`, or no bounds if it never clips `name`."""
    number = r"\s*([-+.\deE]+)\s*"
    match = re.search(rf"clip\(\s*{name}\s*,{number},{number}\)", code)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(rf"clamp\({number},\s*{name}\s*,{number}\)", code)
    return (float(match.group(1)), float(match.group(2))) if match else (-np.inf, np.inf)


def fitted_exponents(names, params, code):
    """Both exponents per cell as the program uses them (clipped to its own bounds), or None."""
    pair = exponent_pair(names)
    if pair is None:
        return None
    return tuple(np.clip(params[:, names.index(name)], *clip_bounds(code, name)) for name in pair)
