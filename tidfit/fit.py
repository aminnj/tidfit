import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches
import matplotlib.lines
import matplotlib.legend

from scipy.optimize import curve_fit
from io import BytesIO
from tokenize import tokenize, NAME
import warnings


class BandObject(matplotlib.patches.Rectangle):
    pass


class BandObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        color = orig_handle.get_facecolor()
        alpha = orig_handle.get_alpha()
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = matplotlib.patches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=color,
            edgecolor="none",
            lw=0.0,
            alpha=alpha,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        patch = matplotlib.lines.Line2D(
            [x0 + width * 0.03, x0 + width - width * 0.05],
            [y0 + height * 0.5],
            color=color,
            alpha=1.0,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


def has_uniform_spacing(obj, epsilon=1e-6):
    offsets = np.ediff1d(obj)
    return np.all(np.abs(offsets - offsets[0]) < epsilon)


def curve_fit_wrapper(func, xdata, ydata, sigma=None, absolute_sigma=False, **kwargs):
    """
    Wrapper around `scipy.optimize.curve_fit`. Initial parameters (`p0`)
    can be set in the function definition with defaults for kwargs
    (e.g., `func = lambda x,a=1.,b=2.: x+a+b`, will feed `p0 = [1.,2.]` to `curve_fit`)
    """

    if func.__defaults__ and len(func.__defaults__) + 1 == func.__code__.co_argcount:
        func.__defaults__ = tuple(1 if p is None else p for p in func.__defaults__)
        if "p0" not in kwargs:
            kwargs["p0"] = func.__defaults__
    popt, pcov = curve_fit(
        func, xdata, ydata, sigma=sigma, absolute_sigma=absolute_sigma, **kwargs,
    )
    return popt, pcov


def expr_to_lambda(expr):
    """
    Converts a string expression like
        "a+b*np.exp(-c*x+math.pi)"
    into a lambda function with 1 variable and N parameters,
        lambda x,a,b,c: "a+b*np.exp(-c*x+math.pi)"
    `x` is assumed to be the main variable.
    Very simple logic that ignores things like `foo.bar`
    or `foo(` from being considered a parameter.
    Parameters
    ----------
    expr : str
    Returns
    -------
    callable/lambda
    """

    varnames = []
    g = list(tokenize(BytesIO(expr.encode("utf-8")).readline))
    for ix, x in enumerate(g):
        toknum = x[0]
        tokval = x[1]
        if toknum != NAME:
            continue
        if ix > 0 and g[ix - 1][1] in ["."]:
            continue
        if ix < len(g) - 1 and g[ix + 1][1] in [".", "("]:
            continue
        varnames.append(tokval)
    varnames = [name for name in varnames if name != "x"]
    varnames = list(
        dict.fromkeys(varnames)
    )  # remove duplicates, preserving order (python>=3.7)
    lambdastr = f"lambda x,{','.join(varnames)}: {expr}"
    return eval(lambdastr)


def fit(
    func,
    xdata,
    ydata,
    w=None,
    sigma=None,
    curve_fit_kwargs=dict(),
    nsamples=200,
    oversamplex=True,
    mask=None,
    ax=None,
    draw=True,
    color="C3",
    legend=True,
    label=r"fit $\pm$1$\sigma$",
):
    r"""
    Fits a function to data via `scipy.optimize.curve_fit`,
    calculating a 1-sigma band, and optionally plotting it.
    Parameters
    ----------
    func : function taking x data as the first argument, followed by parameters, or a string
    xdata : array of x values
    ydata : array of y values
    w: array of weights
    sigma: array of uncertainties on `ydata`
    curve_fit_kwargs : dict
       dict of extra kwargs to pass to `scipy.optimize.curve_fit`
    nsamples : int, default 200
        number of samples/bootstraps for calculating error bands
    oversamplex : bool, default True
        oversample `xdata` for plotting to get a smoother curve
    mask : array of positive booleans for which values to consider in the fit
    ax : matplotlib AxesSubplot object, default None
    draw : bool, default True
       draw to a specified or pre-existing AxesSubplot object
    color : str, default "red"
       color of fit line and error band
    label : str, default r"fit $\pm$1$\sigma$"
       legend entry label. Parameters will be appended unless this
       is empty.
    legend : bool, default True
        draw a legend
    Returns
    -------
    dict of
        - parameter names, values, errors (sqrt of diagonal of the cov. matrix)
        - a callable function "func" corresponding to the input function
          but with fitted parameters
    """

    if isinstance(func, str):
        func = expr_to_lambda(func)

    if (w is not None) and (sigma is not None):
        raise Exception("Weights and sigma cannot be specified simultaneously")

    absolute_sigma = sigma is not None
    if w is not None:
        sigma = 1 / np.asarray(w)

    if mask is None:
        mask = slice(None)

    xdata_raw = np.asarray(xdata)
    ydata_raw = np.asarray(ydata)

    xdata_mask = xdata_raw[mask]
    ydata_mask = ydata_raw[mask]
    if sigma is not None:
        sigma_mask = sigma[mask]
    else:
        sigma_mask = None

    popt, pcov = curve_fit_wrapper(
        func, xdata_mask, ydata_mask, sigma=sigma_mask, absolute_sigma=absolute_sigma,
    )

    class wrapper(dict):
        def _repr_html_(self):
            s = "<table><tr><th>parameter</th><th>value</th></tr>"
            for name, x in self["params"].items():
                s += f"<tr><td>{name}</td><td>{x['value']:.4g} &plusmn; {x['error']:.4g}</td></tr>"
            s += "</table>"
            return s

    parnames = func.__code__.co_varnames[1:]
    parvalues = popt
    parerrors = np.diag(pcov) ** 0.5
    params = dict()
    for name, v, e in zip(parnames, parvalues, parerrors):
        params[name] = dict(value=v, error=e)

    res = wrapper(
        params=params,
        parnames=parnames,
        parvalues=parvalues,
        parerrors=parerrors,
        func=lambda x: func(x, *parvalues),
    )

    if draw:
        if not ax:
            import matplotlib.pyplot as plt

            ax = plt.gca()

        if has_uniform_spacing(xdata_raw):
            xdata_fine = np.linspace(xdata_raw.min(), xdata_raw.max(), len(xdata) * 5)
        else:
            xdata_fine = np.vstack(
                [
                    xdata_raw,
                    xdata_raw + np.concatenate([np.diff(xdata_raw) / 2, [np.nan]]),
                ]
            ).T.flatten()[:-1]
        fit_ydata = func(xdata_fine, *popt)

        if np.isfinite(pcov).all():
            vopts = np.random.multivariate_normal(popt, pcov, nsamples)
            sampled_ydata = np.vstack([func(xdata_fine, *vopt).T for vopt in vopts])
            sampled_stds = np.nanstd(sampled_ydata, axis=0)
        else:
            warnings.warn("Covariance matrix contains nan/inf")
            sampled_stds = np.ones(len(xdata_fine)) * np.nan

        if label:
            for name, x in params.items():
                label += "\n    "
                label += rf"{name} = {x['value']:.3g} $\pm$ {x['error']:.3g}"
        ax.plot(xdata_fine, fit_ydata, color=color, zorder=3)

        ax.fill_between(
            xdata_fine,
            fit_ydata - sampled_stds,
            fit_ydata + sampled_stds,
            facecolor=color,
            alpha=0.25,
            zorder=3,
        )
        matplotlib.legend.Legend.update_default_handler_map(
            {BandObject: BandObjectHandler()}
        )
        ax.add_patch(
            BandObject(
                (0, 0), 0, 0, label=label, color=color, alpha=0.25, visible=False
            )
        )

        if legend:
            ax.legend()

    return res
