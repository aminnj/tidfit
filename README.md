## tidfit

![Python application](https://github.com/aminnj/tidfit/workflows/Python%20application/badge.svg)

```bash
pip install tidfit
```

### Overview

This package provides a tiny fitting routine to fit a curve to pairs of points and draw it
with some error bands. Only depends on `numpy`, `scipy`, and `matplotlib`.


Test data:
```python
x = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
y = np.array([184., 193., 199., 208., 200., 225., 216., 190., 212., 173.])
yerr = y**0.5
```

```python
fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=yerr, fmt="o", label="data")

# Minimal arguments. An eval-able expression needs exactly one `x` to serve as the
# independent variable. The rest are considered as fittable function parameters.
fit("a*x+b", x, y)

# Or specify an explicit function. Default arguments get used as initial `p0` to `curve_fit`
# Either `w` (weights) or `sigma` (y value uncertainties) can be provided
fit((lambda x, a=1, b=2: a * x + b), x, y, sigma=yerr, color="C0")

# External functions are usable
# A boolean mask specifies which points to fit
fit("np.polyval([a,b],x)", x, y, mask=(x < 0.7), color="C1");
```

<img src="https://user-images.githubusercontent.com/5760027/119238627-300a1900-bb09-11eb-87ce-c7ef36190f75.png" width="350px" />

The return value of the `fit` function has a nice representation in notebooks
```python
out = fit("a*x+b", x, y, draw=False)
out
```
parameter | value
-- | --
a | 2.303 ± 18.34
b | 198.8 ± 10.57

but is just a dict at the end of the day, with two ways of getting the parameter names, values, and errors:
``` python
from pprint import pprint
pprint(out)
```

```python
{'func': <function fit.<locals>.<lambda> at 0x1279c6b70>,
 'params': {'a': {'error': 18.33867031688321, 'value': 2.303030386268574},
            'b': {'error': 10.574593995942141, 'value': 198.84848538965056}},
 'parerrors': array([18.33867032, 10.574594  ]),
 'parnames': ('a', 'b'),
 'parvalues': array([  2.30303039, 198.84848539])}
```

And for convenience, it contains the fitted function ready to be called with an array of x-values
```python
func = out["func"]

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, func(x))
```

