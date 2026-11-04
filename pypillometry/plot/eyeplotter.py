from .gaze import GazePlotter
from .pupil import PupilPlotter
from ..units import parse_angle, u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from typing import Optional, List, Tuple
from collections.abc import Iterable
from loguru import logger
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

class EyePlotter(GazePlotter,PupilPlotter):
    """
    Class for plotting combined gaze and pupil data. It provides all methods of
    :class:`~pypillometry.GazePlotter` and :class:`~pypillometry.PupilPlotter`
    and additional plots that combine gaze and pupil data.

    It is usually not created directly; instead, it is available as the `plot`
    attribute of an :class:`~pypillometry.EyeData` object, e.g.
    ``d.plot.plot_pupil_foreshortening_error_surface()``.

    Parameters
    ----------
    obj : GenericEyeData
        the data object to plot (stored as `obj`)
    """
