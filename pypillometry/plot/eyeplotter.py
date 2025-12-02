from ..plot.gazeplotter import GazePlotter
from ..plot.pupilplotter import PupilPlotter
from ..units import parse_angle
import numpy as np
import pylab as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from typing import Optional, List, Tuple
from collections.abc import Iterable
from loguru import logger
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

class EyePlotter(GazePlotter,PupilPlotter):
    def plot_pupil_foreshortening_error_surface(
        self,
        eyes: str | List[str] = [],
        plot_range: tuple = (-np.inf, +np.inf),
        units: str = "sec",
        show_screen: bool = True,
        cmap: str = "jet",
        gridsize: int = 30,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        min_samples: Optional[int] = None,
        smooth: Optional[float] = None,
        limits: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        """
        Plot a heatmap showing average pupil size across x/y gaze positions.
        
        This visualizes how pupil size varies across different gaze positions,
        which can reveal foreshortening effects (pupils appear smaller when
        looking away from the camera) or other spatial artifacts.
        Each eye is plotted in a separate subplot.
        
        Parameters
        ----------
        eyes : str or list, optional
            Which eye(s) to plot: "left", "right", "average", or a list of these.
            Default is [] which plots all available eyes.
        plot_range : tuple, optional
            The time range to include in the analysis. Default is (-np.inf, +np.inf),
            i.e. all data.
        units : str, optional
            Time units for plot_range. Default is "sec".
        show_screen : bool, optional
            Whether to plot the screen limits. Default is True.
        cmap : str, optional
            Colormap to use. Default is "jet".
        gridsize : int, optional
            Number of bins in each direction. Default is 30.
        vmin : float, optional
            Minimum value for color scale. If None, uses data minimum.
        vmax : float, optional
            Maximum value for color scale. If None, uses data maximum.
        min_samples : int, optional
            Minimum number of samples required in a bin to display it. Bins with
            fewer samples will not be shown. If None (default), automatically
            calculates a threshold at the 10th percentile of bin counts, effectively
            displaying only the top 90% of bins by sample count.
        smooth : float, optional
            If provided, applies Gaussian smoothing to the binned surface. The value
            is the sigma of the Gaussian kernel in bin units (e.g., smooth=1 applies
            minimal smoothing with a 1-bin sigma, smooth=2 applies more smoothing).
            When smooth is provided, uses a rectangular grid instead of hexbin.
        limits : tuple, optional
            Axis limits as (xmin, xmax, ymin, ymax). If not provided, uses screen
            limits when available (and data falls within), otherwise uses data extent.
            
        Returns
        -------
        None
            Creates a matplotlib figure with the heatmap(s)
            
        Examples
        --------
        Plot foreshortening effect for left eye:
        
        >>> data.plot.plot_pupil_foreshortening_error_surface('left')
        
        Plot for multiple eyes:
        
        >>> data.plot.plot_pupil_foreshortening_error_surface(['left', 'right'])
        
        Plot all available eyes with custom color range:
        
        >>> data.plot.plot_pupil_foreshortening_error_surface(vmin=3, vmax=5)
        
        Plot with a specific minimum sample threshold:
        
        >>> data.plot.plot_pupil_foreshortening_error_surface(min_samples=50)
        
        Plot with Gaussian smoothing:
        
        >>> data.plot.plot_pupil_foreshortening_error_surface(smooth=1.5)
        
        Plot with custom limits:
        
        >>> data.plot.plot_pupil_foreshortening_error_surface(limits=(0, 1920, 0, 1080))
        """
        obj = self.obj
        
        # Handle eyes parameter
        if len(eyes) == 0:
            eyes = obj.eyes
        if not isinstance(eyes, list):
            eyes = [eyes]
        
        # Handle time range
        fac = obj._unit_fac(units)
        tx = obj.tx * fac
        
        start, end = plot_range
        if start == -np.inf:
            startix = 0
        else:
            startix = np.argmin(np.abs(tx - start))
            
        if end == np.inf:
            endix = tx.size
        else:
            endix = np.argmin(np.abs(tx - end))
        
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
        
        # Plot each eye
        for eye, ax in zip(eyes, axs):
            # Get variable names
            vx = f"{eye}_x"
            vy = f"{eye}_y"
            vpupil = f"{eye}_pupil"
            
            # Check if the data exists for this eye
            if vx not in obj.data.keys() or vy not in obj.data.keys():
                ax.text(0.5, 0.5, f"Gaze data not available\nfor '{eye}' eye", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{eye} eye')
                continue
                
            if vpupil not in obj.data.keys():
                ax.text(0.5, 0.5, f"Pupil data not available\nfor '{eye}' eye", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{eye} eye')
                continue
            
            # Get masked arrays (automatically filters masked data)
            x_masked = obj[vx][startix:endix]
            y_masked = obj[vy][startix:endix]
            pupil_masked = obj[vpupil][startix:endix]
            
            # Combine masks from all three variables
            combined_mask = (
                np.ma.getmaskarray(x_masked) | 
                np.ma.getmaskarray(y_masked) | 
                np.ma.getmaskarray(pupil_masked)
            )
            
            # Extract only non-masked data
            x_plot = x_masked.data[~combined_mask]
            y_plot = y_masked.data[~combined_mask]
            pupil_plot = pupil_masked.data[~combined_mask]
            
            # Check if we have any data to plot
            if len(x_plot) == 0:
                ax.text(0.5, 0.5, f"No non-masked data\navailable for '{eye}' eye", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{eye} eye')
                continue
            
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
                # Use rectangular binning with Gaussian smoothing
                # Compute binned statistics (mean pupil per bin)
                stat, xedges, yedges, binnumber = binned_statistic_2d(
                    x_plot, y_plot, pupil_plot,
                    statistic='mean', 
                    bins=gridsize,
                    range=[[xmin, xmax], [ymin, ymax]]
                )
                # Also get counts to mask bins with insufficient samples
                counts, _, _, _ = binned_statistic_2d(
                    x_plot, y_plot, pupil_plot,
                    statistic='count',
                    bins=gridsize,
                    range=[[xmin, xmax], [ymin, ymax]]
                )
                
                # Determine min_samples threshold for masking
                if min_samples is None:
                    valid_counts = counts[counts > 0]
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
                stat_masked = np.where(counts >= mincnt, stat, np.nan)
                
                # Apply Gaussian smoothing (only to valid values)
                # Create a mask for valid data
                valid_mask = ~np.isnan(stat_masked)
                stat_filled = np.where(valid_mask, stat_masked, 0)
                
                # Smooth the data and the mask separately, then normalize
                smoothed_data = gaussian_filter(stat_filled.T, sigma=smooth)
                smoothed_mask = gaussian_filter(valid_mask.astype(float).T, sigma=smooth)
                
                # Normalize to get proper weighted average, mask where no data
                with np.errstate(divide='ignore', invalid='ignore'):
                    stat_smooth = np.where(smoothed_mask > 0.01, 
                                          smoothed_data / smoothed_mask, 
                                          np.nan)
                
                im = ax.imshow(
                    stat_smooth,
                    origin='lower',
                    extent=[xmin, xmax, ymin, ymax],
                    cmap=cmap,
                    aspect='auto',
                    vmin=vmin,
                    vmax=vmax
                )
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
                
                # Create hexbin plot with average pupil size
                im = ax.hexbin(
                    x_plot, 
                    y_plot, 
                    C=pupil_plot,
                    gridsize=gridsize, 
                    cmap=cmap,
                    reduce_C_function=np.mean,
                    mincnt=mincnt,
                    vmin=vmin,
                    vmax=vmax
                )
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_label('Avg Pupil Size')
            
            # Set title and labels
            ax.set_title(f'{eye} eye')
            ax.set_xlabel('Gaze X Position')
            ax.set_ylabel('Gaze Y Position')
            ax.set_aspect('equal')
            
            # Set axis limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            
            # Plot screen boundaries
            if show_screen and obj.experimental_setup is not None and obj.experimental_setup.has_screen_info():
                setup = obj.experimental_setup
                screenrect = patches.Rectangle(
                    (setup.screen_xlim[0], setup.screen_ylim[0]), 
                    setup.screen_xlim[1] - setup.screen_xlim[0],
                    setup.screen_ylim[1] - setup.screen_ylim[0],
                    fill=False, 
                    edgecolor="red", 
                    linewidth=2
                )
                ax.add_patch(screenrect)
    
    def plot_experimental_setup(
        self,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
        calibration = None,
        show_gaze_samples: bool = True,
        n_gaze_samples: int = 9,
        ax: Optional[plt.Axes] = None,
        viewing_angle: tuple = None,
        projection: str = '2d'
    ) -> tuple:
        """
        Plot visualization of eye-tracking experimental setup geometry.
        
        Shows the spatial relationship between eye, camera, and screen using
        the experimental parameters stored in the EyeData object. Delegates
        to ExperimentalSetup.plot().
        
        Parameters
        ----------
        theta : float, optional
            Camera polar angle in radians. If None, uses setup's theta or calibration's theta.
        phi : float, optional
            Camera azimuthal angle in radians. If None, uses setup's phi or calibration's phi.
        calibration : ForeshorteningCalibration, optional
            Fitted calibration object containing theta and phi.
        show_gaze_samples : bool, default True
            Show sample gaze vectors to screen positions
        n_gaze_samples : int, default 9
            Number of sample gaze positions
        ax : matplotlib axis, optional
            Existing axis to plot on (only for '3d' projection)
        viewing_angle : tuple, optional
            Viewing angle for 3D projection (elev, azim) or (elev, azim, roll) in degrees.
        projection : str, default '2d'
            '2d' for three orthogonal views, '3d' for interactive 3D view
        
        Returns
        -------
        fig : matplotlib Figure
        ax : matplotlib axis or array of axes
        
        Examples
        --------
        >>> # Use camera geometry from setup
        >>> data.plot.plot_experimental_setup()
        >>> 
        >>> # Use fitted angles from calibration
        >>> calib = data.fit_foreshortening(eye='left')
        >>> data.plot.plot_experimental_setup(calibration=calib)
        >>> 
        >>> # Override angles
        >>> data.plot.plot_experimental_setup(theta=np.radians(20), phi=np.radians(-90))
        """
        obj = self.obj
        setup = obj.experimental_setup
        
        if setup is None:
            raise ValueError("experimental_setup is not set on this EyeData object")
        
        # Get theta and phi from calibration if provided
        if calibration is not None:
            if theta is None:
                theta = calibration.theta
                logger.info(f"Using theta={np.degrees(theta):.1f}° from calibration")
            if phi is None:
                phi = calibration.phi
                logger.info(f"Using phi={np.degrees(phi):.1f}° from calibration")
        
        # Parse angles if provided as strings
        if theta is not None:
            theta = parse_angle(theta)
        if phi is not None:
            phi = parse_angle(phi)
        
        # Parse viewing angle if provided
        if viewing_angle is not None:
            viewing_angle = tuple(np.degrees(parse_angle(va)) for va in viewing_angle)
        
        return setup.plot(
            theta=theta,
            phi=phi,
            projection=projection,
            show_gaze_samples=show_gaze_samples,
            n_gaze_samples=n_gaze_samples,
            ax=ax,
            viewing_angle=viewing_angle,
        )
    