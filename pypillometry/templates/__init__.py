"""Jinja templates for the HTML displays and reports of pypillometry.

:func:`render` renders one of the templates in this folder, :func:`sparkline`
draws the small plots they show (both internal)."""
import base64
import functools
from io import BytesIO

import numpy as np
from loguru import logger


@functools.lru_cache(maxsize=None)
def _environment():
    """Jinja environment for the templates in this folder (autoescaped, so that
    names and labels with "<" or "&" are shown as text)."""
    import jinja2
    from ..quality import _bounds_text
    env = jinja2.Environment(loader=jinja2.PackageLoader("pypillometry", "templates"),
                             autoescape=True, trim_blocks=True, lstrip_blocks=True)
    env.filters["bounds"] = _bounds_text
    return env


def render(template: str, **context) -> str:
    """Render the template `template` (a file name in this folder) with
    `context`."""
    return _environment().get_template(template).render(**context)


def _bin_means(values, nbins):
    """Mean of `values` in `nbins` consecutive bins, ignoring NaN (bins without
    data are NaN)."""
    valid = ~np.isnan(values)
    starts = np.linspace(0, len(values), nbins + 1).astype(int)[:-1]
    sums = np.add.reduceat(np.where(valid, values, 0.0), starts)
    counts = np.add.reduceat(valid.astype(int), starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def sparkline(values, x=None, mark_x=None, max_points=200):
    """Small line plot of `values` for the HTML displays, as a PNG data URI.

    Masked and NaN values are left out. Longer data are averaged into
    `max_points` bins, so that the image stays small and sparse data still
    show up. `mark_x` draws a dashed red line at this x (e.g., the event
    onset). Returns None if matplotlib cannot draw the plot.
    """
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        values = np.ma.filled(np.ma.asarray(values, dtype=float), np.nan)
        x = np.arange(len(values), dtype=float) if x is None else np.asarray(x, dtype=float)
        if len(values) == 0:
            return None
        if len(values) > max_points:
            values, x = _bin_means(values, max_points), _bin_means(x, max_points)
        fig = Figure(figsize=(1.8, 0.6), dpi=100)
        ax = fig.subplots()
        ax.plot(x, values, color="#1f77b4", linewidth=1)
        if mark_x is not None:
            ax.axvline(x=mark_x, color="#d62728", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xlim(x[0], x[-1])
        ax.axis("off")
        buf = BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
    except Exception as exc:  # pragma: no cover - optional feature
        logger.debug(f"Could not render sparkline: {exc}")
        return None
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
