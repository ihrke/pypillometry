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
    
    def plot_experimental_setup(
        self,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
        calibration = None,
        show_gaze_samples: bool = True,
        n_gaze_samples: int = 9,
        ax: Optional[plt.Axes] = None,
        viewing_angle: tuple = ("90 deg", "-90 deg")
    ) -> tuple:
        """
        Plot 3D visualization of eye-tracking experimental setup geometry.
        
        Shows the spatial relationship between eye, camera, and screen using
        the experimental parameters stored in the EyeData object. Camera
        angles (theta, phi) relative to the eye-screen axis must be provided 
        explicitly or via a ForeshorteningCalibration object.
        
        Parameters
        ----------
        theta : float, str, or pint.Quantity, optional
            Camera polar angle (from +z axis). Required if
            calibration is not provided.
            - Plain number: assumed to be radians (with warning)
            - String: e.g., "20 degrees", "0.349 radians"
            - Quantity: e.g., 20 * ureg.degree
        phi : float, str, or pint.Quantity, optional
            Camera azimuthal angle (from +x axis in xy-plane).
            Required if calibration is not provided.
            - Plain number: assumed to be radians (with warning)
            - String: e.g., "-90 degrees", "-1.57 radians"
            - Quantity: e.g., -90 * ureg.degree
        calibration : ForeshorteningCalibration, optional
            Fitted calibration object containing theta and phi. If provided,
            theta and phi will be extracted from this object (unless explicitly
            overridden by theta/phi parameters).
        show_gaze_samples : bool, default True
            Show sample gaze vectors to screen positions
        n_gaze_samples : int, default 9
            Number of sample gaze positions (3x3 grid)
        ax : matplotlib 3D axis, optional
            Existing 3D axis to plot on. If None, creates new figure.
        viewing_angle : tuple, optional
            Viewing angle (elev, azim). Default is ("90 deg", "-90 deg").
            - Plain number: assumed to be radians (with warning)
            - String: e.g., "90 degrees", "1.57 radians"
            - Quantity: e.g., 90 * ureg.degree
        
        Returns
        -------
        fig : matplotlib Figure
            Figure object (interactive rotation enabled)
        ax : matplotlib 3D axis
            3D axis object
        
        Raises
        ------
        ValueError
            If camera_eye_distance, screen_eye_distance, or physical_screen_size
            are not set in the EyeData object, or if theta and phi are not 
            provided (either explicitly or via calibration parameter).
        
        Examples
        --------
        >>> # Set experimental parameters first
        >>> data.set_experiment_info(
        ...     camera_eye_distance=600,
        ...     screen_eye_distance=70,
        ...     physical_screen_size=(52, 29)
        ... )
        >>> 
        >>> # Specify camera angles explicitly
        >>> data.plot.plot_experimental_setup(theta=np.radians(20), phi=np.radians(-90))
        >>> 
        >>> # Or use fitted camera geometry from foreshortening calibration
        >>> calib = data.fit_foreshortening(eye='left')
        >>> data.plot.plot_experimental_setup(calibration=calib)
        >>> 
        >>> # Compare different angles
        >>> import numpy as np
        >>> fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, 
        ...                          figsize=(14, 6))
        >>> data.plot.plot_experimental_setup(theta=np.radians(20), phi=np.radians(-90), ax=axes[0])
        >>> data.plot.plot_experimental_setup(theta=np.radians(75), phi=0, ax=axes[1])
        """
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        obj = self.obj
        
        # Get required parameters from object (raises ValueError if not set)
        r = obj.camera_eye_distance  # mm
        d = obj.screen_eye_distance  # mm
        screen_size = obj.physical_screen_dims  # (width, height) in mm
        
        # Parse theta and phi if provided explicitly
        if theta is not None:
            theta = parse_angle(theta)
        if phi is not None:
            phi = parse_angle(phi)
        
        # Get theta and phi from calibration if not provided explicitly
        if calibration is not None:
            if theta is None:
                theta = calibration.theta
                logger.info(f"plot_experimental_setup: Using theta={np.degrees(theta):.1f}° from calibration ({calibration.eye} eye)")
            if phi is None:
                phi = calibration.phi
                logger.info(f"plot_experimental_setup: Using phi={np.degrees(phi):.1f}° from calibration ({calibration.eye} eye)")
        
        # Check that we have both theta and phi
        if theta is None or phi is None:
            missing = []
            if theta is None:
                missing.append("theta")
            if phi is None:
                missing.append("phi")
            
            raise ValueError(
                f"Camera angle(s) {', '.join(missing)} not provided. "
                "Either provide theta and phi explicitly, or pass a ForeshorteningCalibration "
                "object via the calibration parameter."
            )
        
        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Eye at origin
        eye_pos = np.array([0, 0, 0])
        
        # Camera position from spherical coordinates
        camera_pos = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])
        
        # Screen corners
        w, h = screen_size
        screen_corners = np.array([
            [-w/2, -h/2, d],  # Bottom-left
            [w/2, -h/2, d],   # Bottom-right
            [w/2, h/2, d],    # Top-right
            [-w/2, h/2, d],   # Top-left
            [-w/2, -h/2, d],  # Back to start
        ])
        
        # Plot eye
        ax.scatter(*eye_pos, color='blue', s=200, marker='o', 
                   label='Eye (E)', edgecolors='darkblue', linewidths=2, alpha=0.8)
        
        # Plot camera
        ax.scatter(*camera_pos, color='red', s=200, marker='^', 
                   label='Camera (C)', edgecolors='darkred', linewidths=2, alpha=0.8)
        
        # Plot screen
        ax.plot(screen_corners[:, 0], screen_corners[:, 1], screen_corners[:, 2],
                'k-', linewidth=2, label='Screen (S)')
        
        # Fill screen with semi-transparent surface
        screen_surface = [[screen_corners[0], screen_corners[1], 
                           screen_corners[2], screen_corners[3]]]
        ax.add_collection3d(Poly3DCollection(screen_surface, alpha=0.2, 
                                             facecolor='gray', edgecolor='black'))
        
        # Plot eye-to-camera line
        ax.plot([eye_pos[0], camera_pos[0]], 
                [eye_pos[1], camera_pos[1]], 
                [eye_pos[2], camera_pos[2]], 
                'r--', linewidth=2, alpha=0.6, label=f'E-C: {r:.0f}mm')
        
        # Plot eye-to-screen center line (viewing axis)
        screen_center = np.array([0, 0, d])
        ax.plot([eye_pos[0], screen_center[0]], 
                [eye_pos[1], screen_center[1]], 
                [eye_pos[2], screen_center[2]], 
                'g--', linewidth=2, alpha=0.6, label=f'E-S: {d:.0f}mm')
        
        # Show sample gaze vectors
        if show_gaze_samples:
            grid_size = int(np.sqrt(n_gaze_samples))
            x_positions = np.linspace(-w/2, w/2, grid_size)
            y_positions = np.linspace(-h/2, h/2, grid_size)
            
            for x in x_positions:
                for y in y_positions:
                    gaze_point = np.array([x, y, d])
                    ax.plot([eye_pos[0], gaze_point[0]], 
                           [eye_pos[1], gaze_point[1]], 
                           [eye_pos[2], gaze_point[2]], 
                           'c-', linewidth=0.5, alpha=0.3)
                    ax.scatter(*gaze_point, color='cyan', s=20, alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('X (mm, right →)', fontsize=10)
        ax.set_ylabel('Y (mm, up ↑)', fontsize=10)
        ax.set_zlabel('Z (mm, forward →)', fontsize=10)
        
        # Title with geometry info
        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)
        title = f'Eye-Tracking Setup Geometry\n'
        title += f'θ={theta_deg:.1f}°, φ={phi_deg:.1f}°, r={r:.0f}mm, d={d:.0f}mm'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Set equal aspect ratio for better visualization
        max_range = max(r, d, w/2, h/2) * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range/4, d + max_range/4])
        
        # Legend
        ax.legend(loc='upper left', fontsize=9)
        
        # Add coordinate system arrows at origin
        arrow_length = max_range / 4
        ax.quiver(0, 0, 0, arrow_length, 0, 0, color='r', alpha=0.3, 
                  arrow_length_ratio=0.1, linewidth=1.5)
        ax.quiver(0, 0, 0, 0, arrow_length, 0, color='g', alpha=0.3, 
                  arrow_length_ratio=0.1, linewidth=1.5)
        ax.quiver(0, 0, 0, 0, 0, arrow_length, color='b', alpha=0.3, 
                  arrow_length_ratio=0.1, linewidth=1.5)
        
        # Initial viewing angle: behind the eye looking at the screen
        # elev=10 gives slight top-down view, azim=0 aligns with viewing axis
        elev, azim = parse_angle(viewing_angle) # come back in radians
        ax.view_init(elev=np.degrees(elev), azim=np.degrees(azim))
        
        plt.tight_layout()
        
        return fig, ax