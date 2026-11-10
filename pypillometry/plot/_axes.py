"""
Axes and figures of the plotting functions.

All plotting functions of the package follow one of three conventions:

- single-panel plots take ``ax=None``, draw on `ax` (or on the current
  axes) and return the axes (:func:`get_ax`)
- multi-panel plots take ``axes=None`` and ``figsize=None``, draw on the
  given axes (one per panel) or on a new figure and return the figure
  (:func:`get_axes`)
- plots with several figures take ``pdf_file`` and ``figsize`` and return
  the list of figures (:func:`write_pdf`)
"""
from pathlib import Path

import numpy as np
from loguru import logger


def get_ax(ax=None):
    """`ax`, or the current axes if `ax` is None."""
    import matplotlib.pyplot as plt
    return plt.gca() if ax is None else ax


def get_axes(axes, n: int, what: str = "panel", figsize=None,
             nrows: int = None, ncols: int = None, subplot_kw=None):
    """
    The axes of a multi-panel plot: the given `axes` or a new figure.

    Parameters
    ----------
    axes : Axes, sequence of Axes or None
        axes given by the user (one per panel), or None for a new figure
    n : int
        number of panels
    what : str
        what one panel shows (for the error message), e.g. "variable"
    figsize : tuple, optional
        size of the new figure (default: matplotlib's default)
    nrows, ncols : int, optional
        layout of the new figure; default: `n` rows, one column
    subplot_kw : dict, optional
        passed to :meth:`matplotlib.figure.Figure.subplots` (e.g.,
        ``{"projection": "3d"}``)

    Returns
    -------
    fig : Figure
        the figure of the axes
    axes : list of Axes
        `n` axes

    Raises
    ------
    ValueError
        if `axes` does not contain `n` axes
    """
    import matplotlib.pyplot as plt
    if axes is not None:
        axes = list(np.ravel(np.array(axes, dtype=object)))
        if len(axes) != n:
            raise ValueError(f"axes: {n} axes needed (one per {what}), got {len(axes)}")
        return axes[0].figure, axes
    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(nrows or n, ncols or 1, squeeze=False, subplot_kw=subplot_kw)
    return fig, list(axs.ravel())


def write_pdf(figs, pdf_file) -> None:
    """Write `figs` into the multi-page PDF file `pdf_file` and close them
    (missing directories are created)."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = Path(pdf_file)
    if not pdf_path.parent.exists():
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: {}", pdf_path.parent)
    logger.info("Writing PDF file '{}'", pdf_file)
    with PdfPages(pdf_file) as pdf:
        for fig in figs:
            pdf.savefig(fig)
    for fig in figs:
        plt.close(fig)
