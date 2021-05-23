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



if __name__ == "__main__":
    pytest.main(["--capture=no", __file__])
