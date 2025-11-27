from ..plot.gazeplotter import GazePlotter
from ..plot.pupilplotter import PupilPlotter
from ..units import parse_angle
import numpy as np
import pylab as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from typing import Optional, List
from collections.abc import Iterable
from loguru import logger

class EyePlotter(GazePlotter,PupilPlotter):
    def plot_pupil_foreshortening_error_surface(
        self,
        eyes: str | List[str] = [],
        plot_range: tuple = (-np.inf, +np.inf),
        units: str = "sec",
        show_screen: bool = True,
        cmap: str = "viridis",
        gridsize: int = 30,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
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
            Colormap to use. Default is "viridis".
        gridsize : int, optional
            Number of hexagonal bins in each direction. Default is 30.
        vmin : float, optional
            Minimum value for color scale. If None, uses data minimum.
        vmax : float, optional
            Maximum value for color scale. If None, uses data maximum.
            
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
            
            # Create hexbin plot with average pupil size
            im = ax.hexbin(
                x_plot, 
                y_plot, 
                C=pupil_plot,
                gridsize=gridsize, 
                cmap=cmap,
                reduce_C_function=np.mean,
                mincnt=1,
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
    