from ..eyedata import GenericEyeData
from .genplotter import GenericPlotter
import numpy as np
from collections.abc import Iterable
from typing import List, Union, Optional, Tuple

import pylab as plt
import matplotlib.patches as patches
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from ..roi import ROI
from loguru import logger
from scipy.ndimage import gaussian_filter

class GazePlotter(GenericPlotter):
    """
    Class for plotting eye data. The class is initialized with an EyeData object
    and provides methods to plot the data in various ways.
    """
    obj: GenericEyeData # link to the data object

    def __init__(self, obj: GenericEyeData):
        self.ignore_vars = [] #["blinkmask", "pupilinterpolated"]
        self.obj = obj
    
    def plot_heatmap(
        self,
        plot_range: tuple = (-np.inf, +np.inf),
        eyes: list = [],
        show_screen: bool = True,
        show_masked: bool = False,
        rois: List[ROI] = None,
        roi_style: dict={},
        units: str = "sec",
        cmap: str = "jet",
        gridsize: int|str = 30,#"auto"
        min_samples: Optional[int] = 1,
        smooth: Optional[float] = None,
        limits: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        """
        Plot EyeData as a heatmap. Typically used for a large amount of data
        to spot the most frequent locations.
        To plot a scanpath, use :meth:`.plot_scanpath()` instead.
        Each eye is plotted in a separate subplot.

        Parameters:
        -----------

        plot_range: tuple
            The time range to plot. Default is (-np.inf, +np.inf), i.e. all data.
        eyes: list
            The eyes to plot. Default is [], which means all available data ("left", "right",
            average, regression, ...)
        show_screen: bool
            Whether to plot the screen limits. Default is True.
        show_masked: bool
            Whether to plot the masked data (because of blinks, artifacts, ...). Default is False.
        rois: List[ROI], optional
            List of ROIs to plot. Default is None.
        roi_style: dict, optional
            Style for the ROIs. Default is None.
            Example:
            {
                "facecolor": "red",
                "edgecolor": "black",
                "linewidth": 2,
                "alpha": 0.5
            }
        units: str
            The units to plot. Default is "sec".
        cmap: str
            The colormap to use. Default is "jet".
        gridsize: str or int
            The gridsize for the hexbin plot or histogram. Default is 30.
        min_samples: int, optional
            Minimum number of samples required in a bin to display it. Bins with
            fewer samples will not be shown. If None, automatically calculates a 
            threshold at the 10th percentile of bin counts. Default is 1.
        smooth: float, optional
            If provided, applies Gaussian smoothing to the binned histogram. The value
            is the sigma of the Gaussian kernel in bin units (e.g., smooth=1 applies
            minimal smoothing with a 1-bin sigma, smooth=2 applies more smoothing).
            When smooth is provided, uses a rectangular grid instead of hexbin.
        limits: tuple, optional
            Axis limits as (xmin, xmax, ymin, ymax). If not provided, uses screen
            limits when available (and data falls within), otherwise uses data extent.
        """
        obj = self.obj
        
        # Validate plot_range
        self._validate_plot_range(plot_range, units)
        
        fac=obj._unit_fac(units)
        tx=obj.tx*fac

        # plot_range
        start,end=plot_range
        if start==-np.inf:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.inf:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))    
        
        # which eyes to plot
        if len(eyes)==0:
            eyes=obj.eyes
        if not isinstance(eyes, list):
            plot_eyes=[eyes]

        # choose gridsize
        if gridsize=="auto":
            gridsize=int(np.sqrt(endix-startix))

        # Use current figure or create new one if none exists
        nplots = len(eyes)
        fig = plt.gcf()
        
        # Check if current figure is empty/new
        if len(fig.axes) == 0:
            # Create subplots in the current figure
            axs = fig.subplots(1, nplots)
            # Make axs iterable even for single plot
            if not isinstance(axs, Iterable):
                axs = [axs]
        else:
            # Use existing axes
            axs = fig.axes
            if len(axs) != nplots:
                logger.warning(f"Current figure has {len(axs)} axes but {nplots} eyes to plot. Creating new subplots.")
                fig.clear()
                axs = fig.subplots(1, nplots)
                if not isinstance(axs, Iterable):
                    axs = [axs]
        
        for eye, ax in zip(eyes, axs):
            vx = "_".join([eye,"x"])
            vy = "_".join([eye,"y"])
            if vx in obj.data.keys() and vy in obj.data.keys():
                x_data = obj.data[vx][startix:endix]
                y_data = obj.data[vy][startix:endix]
                if not show_masked and vx in obj.data.mask and vy in obj.data.mask:
                    mask = obj.data.mask[vx][startix:endix] | obj.data.mask[vy][startix:endix]
                    mask = mask.astype(bool)
                    x_plot = x_data[~mask]
                    y_plot = y_data[~mask]
                else:
                    x_plot = x_data
                    y_plot = y_data
                
                # Determine limits
                if limits is not None:
                    xmin, xmax, ymin, ymax = limits
                else:
                    # Use screen limits if available and data falls within, else use data extent
                    if obj.experimental_setup is not None and obj.experimental_setup.has_screen_info():
                        setup = obj.experimental_setup
                        xmin = max(x_plot.min(), setup.screen_xlim[0])
                        xmax = min(x_plot.max(), setup.screen_xlim[1])
                        ymin = max(y_plot.min(), setup.screen_ylim[0])
                        ymax = min(y_plot.max(), setup.screen_ylim[1])
                    else:
                        xmin, xmax = x_plot.min(), x_plot.max()
                        ymin, ymax = y_plot.min(), y_plot.max()
                
                if smooth is not None:
                    # Use rectangular histogram with Gaussian smoothing
                    hist, xedges, yedges = np.histogram2d(
                        x_plot, y_plot,
                        bins=gridsize,
                        range=[[xmin, xmax], [ymin, ymax]]
                    )
                    
                    # Determine min_samples threshold for masking
                    if min_samples is None:
                        valid_counts = hist[hist > 0]
                        if len(valid_counts) > 0:
                            mincnt = int(np.percentile(valid_counts, 10))
                            mincnt = max(1, mincnt)
                            logger.info(f"[{eye} eye] Auto-calculated min_samples threshold: {mincnt} "
                                       f"(10th percentile of {len(valid_counts)} bins with counts ranging "
                                       f"{int(valid_counts.min())}-{int(valid_counts.max())})")
                        else:
                            mincnt = 1
                    else:
                        mincnt = min_samples
                    
                    # Mask bins with insufficient samples
                    hist_masked = np.where(hist >= mincnt, hist, np.nan)
                    
                    # Apply Gaussian smoothing (only to valid values)
                    valid_mask = ~np.isnan(hist_masked)
                    hist_filled = np.where(valid_mask, hist_masked, 0)
                    
                    # Smooth the data and the mask separately, then normalize
                    smoothed_data = gaussian_filter(hist_filled.T, sigma=smooth)
                    smoothed_mask = gaussian_filter(valid_mask.astype(float).T, sigma=smooth)
                    
                    # Normalize to get proper weighted average, mask where no data
                    with np.errstate(divide='ignore', invalid='ignore'):
                        hist_smooth = np.where(smoothed_mask > 0.01, 
                                              smoothed_data / smoothed_mask, 
                                              np.nan)
                    
                    im = ax.imshow(
                        hist_smooth,
                        origin='lower',
                        extent=[xmin, xmax, ymin, ymax],
                        cmap=cmap,
                        aspect='auto'
                    )
                    
                    # Set axis limits
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                else:
                    # Determine min_samples threshold
                    if min_samples is None:
                        # Create temporary hexbin to get bin counts
                        temp_hexbin = ax.hexbin(
                            x_plot, 
                            y_plot, 
                            gridsize=gridsize, 
                            mincnt=1
                        )
                        counts = temp_hexbin.get_array()
                        # Use 10th percentile as threshold (keeps top 90%)
                        mincnt = int(np.percentile(counts, 10))
                        mincnt = max(1, mincnt)  # Ensure at least 1
                        logger.info(f"[{eye} eye] Auto-calculated min_samples threshold: {mincnt} "
                                   f"(10th percentile of {len(counts)} bins with counts ranging {int(counts.min())}-{int(counts.max())})")
                        # Clear the temporary plot
                        ax.clear()
                    else:
                        mincnt = min_samples
                    
                    im = ax.hexbin(x_plot, y_plot, gridsize=gridsize, cmap=cmap, mincnt=mincnt)
                    
                    # Set axis limits for hexbin too
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_title(eye)
            ax.set_aspect("equal")
            #fig.colorbar()
            if show_screen and obj.experimental_setup is not None:
                setup = obj.experimental_setup
                screenrect=patches.Rectangle(
                    (setup.screen_xlim[0], setup.screen_ylim[0]), 
                    setup.screen_xlim[1] - setup.screen_xlim[0],
                    setup.screen_ylim[1] - setup.screen_ylim[0], 
                    fill=False, edgecolor="red", linewidth=2)
                ax.add_patch(screenrect)
            if rois is not None:
                for roi in rois:
                    roi.plot(ax, **roi_style)

    def plot_scanpath(self, 
            plot_range: tuple=(-np.inf, +np.inf), 
            eyes: list=[],
            show_screen: bool=True,
            rois: List[ROI]=None,
            roi_style: dict={},
            show_onsets: bool=True,
            title: str="",
            units: str=None,
            figsize: tuple=(10,10)
            ) -> None:
        """
        Plot EyeData as a scanpath. Typically used for single trials
        or another small amount of data. To plot a larger amount of data,
        use :meth:`.plot_heatmap()` instead.

        If several eyes are available, they are plotted in the same figure 
        in different colors. You can choose between them using the `plot_eyes` 
        parameter.

        Parameters:
        -----------

        plot_range: tuple
            The time range to plot. Default is (-np.inf, +np.inf), i.e. all data.
        eyes: list
            The eyes to plot. Default is [], which means all available data ("left", "right",
            average, regression, ...)
        show_screen: bool
            Whether to plot the screen limits. Default is True.
        rois: List[ROI], optional
            List of ROIs to plot. Default is None.
        roi_style: dict, optional
            Style for the ROIs. Default is None.
            Example:
            {
                "facecolor": "red",
                "edgecolor": "black",
                "linewidth": 2,
                "alpha": 0.5
            }
        show_onsets: bool
            Whether to plot the event onsets. Default is True.
        title: str
            The title of the plot. Default is "".
        units: str
            The units to plot. Default is "sec". Plotted only for the first "eye".
            If None, the units are taken from the data.
        figsize: tuple
            The figure size (per subplot). Default is (10,5).
        """
        obj = self.obj

        # Validate plot_range
        self._validate_plot_range(plot_range, units)

        if units is not None: 
            fac=obj._unit_fac(units)
            tx = obj.tx*fac
            evon = obj.event_onsets*fac
        else:
            tx=obj.tx.copy()
            evon=obj.event_onsets.copy()

        # plot_range
        start,end=plot_range
        if start==-np.inf:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.inf:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))    
        
        # get events in range
        evonixx=np.logical_and(evon>=start, evon<end)
        evlab=obj.event_labels[evonixx]
        evon=evon[evonixx]
        evontix=np.argmin(np.abs(tx-evon[:,np.newaxis]), axis=1).astype(int)

        # which eyes to plot
        if len(eyes)==0:
            eyes=obj.eyes

        fig, ax = plt.subplots(1,1)
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])

        tnorm = tx[startix:endix].copy()
        tnorm = (tnorm - tnorm.min()) / (tnorm.max() - tnorm.min())
        for eye in eyes:
            vx = "_".join([eye,"x"])
            vy = "_".join([eye,"y"])
            if vx in obj.data.keys() and vy in obj.data.keys():
                ax.plot(obj.data[vx][startix:endix], obj.data[vy][startix:endix], alpha=0.3, label=eye)
                ax.scatter(obj.data[vx][startix:endix], obj.data[vy][startix:endix], 
                        s=1, c=cm.jet(tnorm))
            if show_onsets and eye==eyes[0]:
                for ix, lab in zip(evontix, evlab):
                    ax.text(obj.data[vx][ix], obj.data[vy][ix], lab, 
                            fontsize=12, ha="center", va="center")

        ax.set_aspect("equal")
        #fig.colorbar()
        if show_screen and obj.experimental_setup is not None:
            setup = obj.experimental_setup
            screenrect=patches.Rectangle(
                (setup.screen_xlim[0], setup.screen_ylim[0]), 
                setup.screen_xlim[1] - setup.screen_xlim[0],
                setup.screen_ylim[1] - setup.screen_ylim[0], 
                fill=False, edgecolor="red", linewidth=2)
            ax.add_patch(screenrect)
        if rois is not None:
            for roi in rois:
                roi.plot(ax, **roi_style)
        ax.legend()
        ax.set_title(title)
        fig.tight_layout()
    
    def plot_calibration(self, eyes: Union[str, list, None] = None, show_surface: bool = True, 
                        interpolation: str = 'rbf', figsize: tuple = (10, 8)):
        """
        Plot spatial calibration data for one or more eyes in subplots.
        
        Creates a figure with subplots for each eye and calls the plot() method 
        on each SpatialCalibration object to show calibration accuracy with 
        target points (black X), measured points (dark blue +), and optional 
        interpolated error surface.
        
        Parameters
        ----------
        eyes : str, list, or None, default None
            Which eye(s) to plot. If None or empty list, plots all available eyes.
            Can be a single eye name (e.g., 'left') or list of eye names (e.g., ['left', 'right']).
        show_surface : bool, default True
            If True, show interpolated error surface as background.
            If False, show only the calibration points.
        interpolation : str, default 'rbf'
            Interpolation method for surface (only used if show_surface=True).
            Options: 'rbf', 'linear', 'cubic', 'nearest'
        figsize : tuple, default (10, 8)
            Figure size per subplot (width, height) in inches.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        axes : list or matplotlib.axes.Axes
            List of axis objects (or single axis if only one eye)
        
        Raises
        ------
        ValueError
            If no calibration data is available
        
        Examples
        --------
        >>> import pypillometry as pp
        >>> data = pp.EyeData.from_eyelink('data.edf')  # doctest: +SKIP
        >>> # Plot all eyes with surface
        >>> fig, axes = data.plot.plot_calibration()  # doctest: +SKIP
        >>> 
        >>> # Plot only left eye without surface
        >>> fig, ax = data.plot.plot_calibration(eyes='left', show_surface=False)  # doctest: +SKIP
        >>> 
        >>> # Plot specific eyes
        >>> fig, axes = data.plot.plot_calibration(eyes=['left', 'right'])  # doctest: +SKIP
        """
        obj = self.obj
        
        # Check if calibration data exists
        if not hasattr(obj, 'calibration') or obj.calibration is None:
            raise ValueError("No calibration data available. Load data with calibration or add it manually.")
        
        # Determine which eyes to plot
        if eyes is None or (isinstance(eyes, list) and len(eyes) == 0):
            eyes_to_plot = list(obj.calibration.keys())
        elif isinstance(eyes, str):
            eyes_to_plot = [eyes]
        else:
            eyes_to_plot = eyes
        
        # Filter to only available eyes
        available_eyes = [eye for eye in eyes_to_plot if eye in obj.calibration]
        
        if len(available_eyes) == 0:
            raise ValueError(f"No calibration data available for requested eyes: {eyes_to_plot}")
        
        # Create figure and subplots
        n_eyes = len(available_eyes)
        fig, axes_array = plt.subplots(1, n_eyes, figsize=(figsize[0] * n_eyes, figsize[1]))
        
        # Make axes iterable even if only one subplot
        if n_eyes == 1:
            axes_array = [axes_array]
        
        # Plot each eye by setting it as current axes and calling plot()
        for ax, eye in zip(axes_array, available_eyes):
            plt.sca(ax)  # Set current axes
            obj.calibration[eye].plot(show_surface=show_surface, interpolation=interpolation)
        
        fig.tight_layout()
        
        # Return single axis if only one eye, otherwise return list
        if n_eyes == 1:
            return fig, axes_array[0]
        else:
            return fig, axes_array    