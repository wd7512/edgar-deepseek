"""Find a learned program's two peak exponents and the range the program itself clips them to."""

import logging
import re

import numpy as np

PAIRS = (
    ("exponent_1", "exponent_2"),
    ("shape_power_1", "shape_power_2"),
    ("power_1", "power_2"),
    ("shape_1", "shape_2"),
)
AMPLITUDES = (("amplitude_1", "amplitude_2"), ("amp_1", "amp_2"), ("A1", "A2"), ("a1", "a2"))
NUMBER = r"\s*(?:(?:a_)?(?:min|max)\s*=\s*)?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*"


def _pair(names, pairs):
    return next((pair for pair in pairs if pair[0] in names and pair[1] in names), None)


def exponent_pair(names):
    """The names of the two peak exponents among a program's parameter names, or None."""
    return _pair(names, PAIRS)


def exponent_groups(names):
    """Each peak's exponent names: one per peak, or a left and a right one per peak. None if neither."""
    pair = exponent_pair(names)
    if pair is not None:
        return (pair[0],), (pair[1],)
    for stem in ("shape", "power", "exponent"):
        groups = tuple((f"{stem}_{k}_left", f"{stem}_{k}_right") for k in (1, 2))
        if all(name in names for group in groups for name in group):
            return groups
    return None


def amplitude_pair(names):
    """The names of the two peak amplitudes among a program's parameter names, or None."""
    return _pair(names, AMPLITUDES)


def clip_bounds(code, name):
    """The plain-number `(low, high)` a program passes to `clip(name, low, high)` or `clamp(low, name, high)`.

    Keyword bounds (`min=`, `a_min=`) count. Anything else, such as an expression or a named
    constant, gives no bounds and a warning.
    """
    for pattern in (rf"clip\(\s*{name}\s*,{NUMBER},{NUMBER}\)", rf"clamp\({NUMBER},\s*{name}\s*,{NUMBER}\)"):
        match = re.search(pattern, code)
        if match:
            return float(match.group(1)), float(match.group(2))
    logging.warning("No plain-number clip found for %s; treating it as unbounded", name)
    return -np.inf, np.inf


def fitted_exponents(names, params, code):
    """Each peak's exponent per cell as the program uses it (clipped to its own bounds), or None.

    A peak with a separate exponent on each side counts its sharper (lower) side, which sets the
    shape of its tip.
    """
    groups = exponent_groups(names)
    if groups is None:
        return None
    return tuple(
        np.min([np.clip(params[:, names.index(name)], *clip_bounds(code, name)) for name in group], axis=0)
        for group in groups
    )
