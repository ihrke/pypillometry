from ..plot.gazeplotter import GazePlotter
from ..plot.pupilplotter import PupilPlotter
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
            if show_screen and hasattr(obj, 'screen_xlim') and hasattr(obj, 'screen_ylim'):
                screenrect = patches.Rectangle(
                    (obj.screen_xlim[0], obj.screen_ylim[0]), 
                    obj.screen_xlim[1] - obj.screen_xlim[0],
                    obj.screen_ylim[1] - obj.screen_ylim[0],
                    fill=False, 
                    edgecolor="red", 
                    linewidth=2
                )
                ax.add_patch(screenrect)