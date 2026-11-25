"""The fixed double-Gaussian seed shared by frontier runs and their viewer."""

BASELINE_ID = "double_gaussian"
BASE = "b + a1*exp(-0.5*(wrap(theta-m)/s)**2) + a2*exp(-0.5*(wrap(theta-m-pi)/s)**2)"


def baseline_node():
    """Return the starting graph node, with its equation but no probe results."""
    return {
        "node": BASELINE_ID,
        "label": "Double-Gaussian tuning curve",
        "text": "Two wrapped Gaussian peaks 180° apart, with one shared width and a constant baseline.",
        "routes": [],
        "introduced_at": 0,
        "equation": BASE,
    }
