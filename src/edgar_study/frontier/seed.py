"""The fixed double-Gaussian seed shared by frontier runs and their viewer."""

BASELINE_ID = "double_gaussian"
BASE = "b + a1*exp(-0.5*(wrap(theta-m)/s)**2) + a2*exp(-0.5*(wrap(theta-m-pi)/s)**2)"


BASE_NAMES = ("m", "b", "a1", "a2", "s")
BASE_PROBES = ((0.73, 0.24, 2.1, 0.4, 0.38), (3.8, 0.36, 0.8, 1.7, 0.91), (5.1, 1.4, 1.2, 2.4, 0.22))


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
