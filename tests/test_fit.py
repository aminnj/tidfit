import numpy as np

import os

from tidfit.fit import fit, expr_to_lambda

np.set_printoptions(linewidth=120)

import pytest


def allclose(a, b, equal_nan=False):
    return np.testing.assert_allclose(np.array(a), np.array(b), equal_nan=equal_nan)


def test_expr_to_lambda():
    f = expr_to_lambda("x+a+a+b+np.pi")
    g = lambda x, a, b: x + a + a + b + np.pi
    assert f(1, 2, 3) == g(1, 2, 3)

    f = expr_to_lambda("m*x+b")
    g = lambda x, m, b: m * x + b
    assert f(1, 2, 3) == g(1, 2, 3)

    f = expr_to_lambda("1+np.poly1d([a,b,c])(x)")
    g = lambda x, a, b, c: 1 + np.poly1d([a, b, c])(x)
    assert f(1, 2, 3, 4) == g(1, 2, 3, 4)

    f = expr_to_lambda("const + norm*np.exp(-(x-mu)**2/(2*sigma**2))")
    g = lambda x, const, norm, mu, sigma: const + norm * np.exp(
        -((x - mu) ** 2) / (2 * sigma ** 2)
    )
    assert f(1, 2, 3, 4, 5) == g(1, 2, 3, 4, 5)


def test_linear_fit():
    np.random.seed(42)
    for (slope, intercept) in [
        (1, 5),
        (1, -5),
        (-10, -10),
        (-10, 5),
        (10, -10),
        (10, 5),
    ]:
        N = 50
        x = np.arange(N)
        y = (slope * x + intercept) + np.random.normal(0, 1, N)

        out = fit("slope*x+intercept", x, y, draw=False)

        dslope = (out["params"]["slope"]["value"] - slope) / slope
        dintercept = (out["params"]["intercept"]["value"] - intercept) / intercept
        assert abs(dslope) < 0.1
        assert abs(dintercept) < 0.1


def test_return_func():
    x = np.arange(5)
    y = np.ones(len(x))
    out = fit("a*x+b", x, y, draw=False)
    func = out["func"]
    allclose(y, func(x))


def test_gaussian():
    x = np.array([-0.101, 0.101, 0.303])
    y = np.array([2, 4, 1])

    gaussian = "constant * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))"
    out = fit(gaussian, x, y, sigma=y ** 0.5, mask=(y > 0), draw=False)
    params = out["params"]

    assert abs(params["constant"]["value"] - 4.1175) < 1e-3
    assert abs(params["mean"]["value"] - 0.0673) < 1e-3
    assert abs(params["sigma"]["value"] - 0.1401) < 1e-3
    assert abs(params["constant"]["error"] - 2.0420) < 1e-3
    assert abs(params["mean"]["error"] - 0.0584) < 1e-3
    assert abs(params["sigma"]["error"] - 0.0531) < 1e-3


def test_mask():
    np.random.seed(42)
    x = np.arange(10)
    y = np.concatenate([1 * np.ones(5), 2 * np.ones(5)])
    y += np.random.normal(0, 0.01, 10)

    assert fit("a+b*x", x, y, mask=(y < 1.5), draw=False)["params"]["a"]["value"] < 1.5
    assert fit("a+b*x", x, y, mask=(y > 1.5), draw=False)["params"]["a"]["value"] > 1.5


if __name__ == "__main__":
    pytest.main(["--capture=no", __file__])
