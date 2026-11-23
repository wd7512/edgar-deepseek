import numpy as np
import pytest

from edgar_study import curves, curves_numpy


def _cells(n=6, trials=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, (n, trials))
    return x


@pytest.mark.parametrize("name", ["double-gaussian", "winner"])
def test_model_matches_the_jax_curve(name):
    x = _cells()
    curve = next(c for c in curves.CURVES if c.name == name)
    params = np.stack(
        [curve.estimator(x[i], np.linspace(5, 20, x.shape[1])) for i in range(x.shape[0])]
    ).astype(np.float32)
    expected = np.stack([np.asarray(curve.model(x[i], *params[i])) for i in range(x.shape[0])])
    got = curves_numpy.model(name, x, params)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["double-gaussian", "winner"])
def test_jacobian_matches_jax_forward_mode(name):
    import jax

    x = _cells()
    curve = next(c for c in curves.CURVES if c.name == name)
    params = np.stack(
        [curve.estimator(x[i], np.linspace(5, 20, x.shape[1])) for i in range(x.shape[0])]
    ).astype(np.float32)

    def single(i):
        return np.asarray(jax.jacfwd(lambda p: curve.model(x[i], *p))(params[i]))

    expected = np.stack([single(i) for i in range(x.shape[0])])
    got = curves_numpy.jacobian(name, x, params)
    np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-4)


def test_shapes_and_parameter_counts():
    x = _cells(n=3, trials=7)
    assert curves_numpy.model("double-gaussian", x, np.ones((3, 5))).shape == (3, 7)
    assert curves_numpy.jacobian("double-gaussian", x, np.ones((3, 5))).shape == (3, 7, 5)
    assert curves_numpy.jacobian("winner", x, np.ones((3, 11))).shape == (3, 7, 11)
