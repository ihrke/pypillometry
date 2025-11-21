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
        viewing_angle: tuple = ("-49 deg", "50 deg", "45 deg"),
        projection: str = '2d'
    ) -> tuple:
        """
        Plot visualization of eye-tracking experimental setup geometry.
        
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
            Viewing angle (elev, azim) for 3D projection. Default is ("90 deg", "-90 deg").
            - Plain number: assumed to be radians (with warning)
            - String: e.g., "90 degrees", "1.57 radians"
            - Quantity: e.g., 90 * ureg.degree
        projection : str, default '2d'
            Type of projection:
            - '3d': Interactive 3D view with rotation
            - '2d': Three orthogonal 2D projections (x-y, x-z, y-z planes)
        
        Returns
        -------
        fig : matplotlib Figure
            Figure object
        ax : matplotlib axis or array of axes
            Single 3D axis (if projection='3d') or array of 3 2D axes (if projection='2d')
        
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
        >>> 
        >>> # 2D orthogonal projections
        >>> data.plot.plot_experimental_setup(calibration=calib, projection='2d')
        """
        # Route to appropriate projection method
        if projection.lower() == '2d':
            return self._plot_experimental_setup_2d(
                theta, phi, calibration, show_gaze_samples, n_gaze_samples
            )
        elif projection.lower() == '3d':
            return self._plot_experimental_setup_3d(
                theta, phi, calibration, show_gaze_samples, n_gaze_samples, ax, viewing_angle
            )
        else:
            raise ValueError(f"projection must be '3d' or '2d', got '{projection}'")
    
    def _plot_experimental_setup_3d(
        self,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
        calibration = None,
        show_gaze_samples: bool = True,
        n_gaze_samples: int = 9,
        ax: Optional[plt.Axes] = None,
        viewing_angle: tuple = ("90 deg", "-90 deg")
    ) -> tuple:
        """3D projection helper for plot_experimental_setup."""
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
        
        # Draw angle annotations
        # Theta: angle between z-axis and eye-to-camera vector
        arc_radius = r * 0.3
        n_arc_points = 20
        
        # Theta arc: in the plane containing both z-axis and camera vector
        # We'll draw it in the vertical plane that contains the camera
        theta_angles = np.linspace(0, theta, n_arc_points)
        # Project camera position onto x-y plane to get the azimuthal direction
        camera_xy_norm = np.linalg.norm(camera_pos[:2])
        if camera_xy_norm > 0:
            # Direction in x-y plane
            phi_unit = camera_pos[:2] / camera_xy_norm
        else:
            phi_unit = np.array([1, 0])  # Default to x-axis if camera on z-axis
        
        theta_arc_x = arc_radius * np.sin(theta_angles) * phi_unit[0]
        theta_arc_y = arc_radius * np.sin(theta_angles) * phi_unit[1]
        theta_arc_z = arc_radius * np.cos(theta_angles)
        
        ax.plot(theta_arc_x, theta_arc_y, theta_arc_z, 
                'orange', linewidth=2.5, alpha=0.8)
        
        # Theta label at midpoint of arc
        mid_theta = theta / 2
        label_r = arc_radius * 1.3
        theta_label_pos = np.array([
            label_r * np.sin(mid_theta) * phi_unit[0],
            label_r * np.sin(mid_theta) * phi_unit[1],
            label_r * np.cos(mid_theta)
        ])
        ax.text(theta_label_pos[0], theta_label_pos[1], theta_label_pos[2],
                f'θ={np.degrees(theta):.1f}°', fontsize=11, color='orange',
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='orange', alpha=0.8))
        
        # Phi: angle in x-y plane from x-axis
        # Only draw if camera is not on z-axis
        if camera_xy_norm > 1e-6:  # Avoid numerical issues
            phi_angles = np.linspace(0, phi, n_arc_points)
            phi_arc_radius = arc_radius * 0.8
            phi_arc_x = phi_arc_radius * np.cos(phi_angles)
            phi_arc_y = phi_arc_radius * np.sin(phi_angles)
            phi_arc_z = np.zeros(n_arc_points)
            
            ax.plot(phi_arc_x, phi_arc_y, phi_arc_z,
                    'purple', linewidth=2.5, alpha=0.8)
            
            # Phi label
            mid_phi = phi / 2
            phi_label_r = phi_arc_radius * 1.4
            phi_label_pos = np.array([
                phi_label_r * np.cos(mid_phi),
                phi_label_r * np.sin(mid_phi),
                0
            ])
            ax.text(phi_label_pos[0], phi_label_pos[1], phi_label_pos[2],
                    f'φ={np.degrees(phi):.1f}°', fontsize=11, color='purple',
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='purple', alpha=0.8))
        
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
        # Support both 2-tuple (elev, azim) and 3-tuple (elev, azim, roll)
        parsed_angles = tuple(parse_angle(va) for va in viewing_angle)
        if len(parsed_angles) == 2:
            elev, azim = parsed_angles
            ax.view_init(elev=np.degrees(elev), azim=np.degrees(azim))
        elif len(parsed_angles) == 3:
            elev, azim, roll = parsed_angles
            ax.view_init(elev=np.degrees(elev), azim=np.degrees(azim), roll=np.degrees(roll))
        else:
            raise ValueError(f"viewing_angle must be a 2-tuple (elev, azim) or 3-tuple (elev, azim, roll), got {len(parsed_angles)} values")
        
        plt.tight_layout()
        
        return fig, ax
    
    def _plot_experimental_setup_2d(
        self,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
        calibration = None,
        show_gaze_samples: bool = True,
        n_gaze_samples: int = 9
    ) -> tuple:
        """2D orthogonal projections helper for plot_experimental_setup."""
        import matplotlib.patches as mpatches
        
        obj = self.obj
        
        # Get required parameters from object
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
            if phi is None:
                phi = calibration.phi
        
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
        
        # Calculate positions
        E = np.array([0, 0, 0])  # Eye at origin
        C = np.array([  # Camera position from spherical coordinates
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])
        
        # Screen corners
        w, h = screen_size
        screen_corners = np.array([
            [-w/2, -h/2, d],  # bottom-left
            [ w/2, -h/2, d],  # bottom-right
            [ w/2,  h/2, d],  # top-right
            [-w/2,  h/2, d],  # top-left
            [-w/2, -h/2, d],  # close the rectangle
        ])
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Icon sizes
        eye_radius = max(r, d, w/2, h/2) * 0.03
        cam_size = eye_radius * 1.5
        
        # --- Subplot 1: x-y plane (top view) ---
        ax = axes[0]
        ax.set_aspect('equal')
        ax.axhline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
        
        # Screen (x-y projection)
        ax.plot(screen_corners[:, 0], screen_corners[:, 1], 'k-', linewidth=2, label='Screen')
        ax.fill(screen_corners[:-1, 0], screen_corners[:-1, 1], 
                color='lightgray', alpha=0.3)
        
        # Eye (pictogram: circle with pupil)
        eye_circle = plt.Circle((E[0], E[1]), eye_radius, color='steelblue', 
                                ec='black', linewidth=2, zorder=10)
        ax.add_patch(eye_circle)
        ax.plot(E[0], E[1], 'k.', markersize=6, zorder=11)  # pupil
        ax.text(E[0], E[1] - eye_radius*1.5, 'E', ha='center', va='top', 
                fontsize=10, fontweight='bold')
        
        # Camera (pictogram: rectangle with lens)
        cam_rect = mpatches.FancyBboxPatch(
            (C[0]-cam_size*0.6, C[1]-cam_size*0.4), cam_size*1.2, cam_size*0.8,
            boxstyle="round,pad=0.05", 
            ec='black', fc='darkred', linewidth=2, zorder=10, alpha=0.7
        )
        ax.add_patch(cam_rect)
        ax.plot(C[0], C[1], 'wo', markersize=8, zorder=11)  # lens
        ax.text(C[0], C[1] - cam_size*0.7, 'C', ha='center', va='top',
                fontsize=10, fontweight='bold', color='white')
        
        # Eye-camera line
        ax.plot([E[0], C[0]], [E[1], C[1]], 'r--', alpha=0.5, linewidth=1.5)
        
        # Phi angle annotation (azimuthal angle in x-y plane)
        # Only draw if camera has non-zero x-y projection
        camera_xy_dist = np.sqrt(C[0]**2 + C[1]**2)
        if camera_xy_dist > 1e-6:
            from matplotlib.patches import Arc
            arc_radius = min(camera_xy_dist, max(w, h)) * 0.3
            # Arc from x-axis (0°) to phi
            # Ensure theta1 < theta2 for correct arc direction
            phi_deg = np.degrees(phi)
            theta1, theta2 = (0, phi_deg) if phi_deg > 0 else (phi_deg, 0)
            arc = Arc((E[0], E[1]), 2*arc_radius, 2*arc_radius, 
                     angle=0, theta1=theta1, theta2=theta2,
                     color='purple', linewidth=2.5, zorder=5)
            ax.add_patch(arc)
            
            # Phi label
            mid_phi = phi / 2
            label_r = arc_radius * 1.3
            phi_label_x = E[0] + label_r * np.cos(mid_phi)
            phi_label_y = E[1] + label_r * np.sin(mid_phi)
            ax.text(phi_label_x, phi_label_y, f'φ={phi_deg:.1f}°',
                   fontsize=10, color='purple', fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='purple', alpha=0.9))
        
        # Gaze samples
        if show_gaze_samples:
            n = int(np.sqrt(n_gaze_samples))
            x_positions = np.linspace(-w/2*0.8, w/2*0.8, n)
            y_positions = np.linspace(-h/2*0.8, h/2*0.8, n)
            for x_pos in x_positions:
                for y_pos in y_positions:
                    ax.plot([E[0], x_pos], [E[1], y_pos], 
                           'g-', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('x (mm)', fontsize=10)
        ax.set_ylabel('y (mm)', fontsize=10)
        ax.set_title('Front View (x-y plane)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        # --- Subplot 2: x-z plane (side view from y-axis) ---
        ax = axes[1]
        ax.set_aspect('equal')
        ax.axhline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
        
        # Screen (z-x projection, rotated: z horizontal, x vertical)
        screen_zx = screen_corners[:, [2, 0]]  # Swap to [z, x]
        ax.plot(screen_zx[:, 0], screen_zx[:, 1], 'k-', linewidth=2, label='Screen')
        
        # Eye (swap coordinates: z on x-axis, x on y-axis)
        eye_circle = plt.Circle((E[2], E[0]), eye_radius, color='steelblue', 
                                ec='black', linewidth=2, zorder=10)
        ax.add_patch(eye_circle)
        ax.plot(E[2], E[0], 'k.', markersize=6, zorder=11)
        ax.text(E[2] - eye_radius*1.5, E[0], 'E', ha='right', va='center',
                fontsize=10, fontweight='bold')
        
        # Camera (swap coordinates)
        cam_rect = mpatches.FancyBboxPatch(
            (C[2]-cam_size*0.4, C[0]-cam_size*0.6), cam_size*0.8, cam_size*1.2,
            boxstyle="round,pad=0.05",
            ec='black', fc='darkred', linewidth=2, zorder=10, alpha=0.7
        )
        ax.add_patch(cam_rect)
        ax.plot(C[2], C[0], 'wo', markersize=8, zorder=11)
        ax.text(C[2], C[0] - cam_size*0.7, 'C', ha='center', va='top',
                fontsize=10, fontweight='bold', color='white')
        
        # Eye-camera line (swap coordinates)
        ax.plot([E[2], C[2]], [E[0], C[0]], 'r--', alpha=0.5, linewidth=1.5,
                label=f'E-C: {r:.0f}mm')
        
        # Eye-screen line (swap coordinates)
        ax.plot([E[2], d], [E[0], 0], 'g--', alpha=0.5, linewidth=1.5,
                label=f'E-S: {d:.0f}mm')
        
        # Theta angle annotation (in z-x plane, showing x-component)
        # This shows the projection of theta onto the z-x plane
        from matplotlib.patches import Arc
        # Calculate the angle in the z-x plane
        theta_zx = np.arctan2(C[0], C[2])  # angle from z-axis in z-x plane
        if abs(theta_zx) > 1e-6:
            arc_radius = min(r, d) * 0.3
            # Arc from z-axis (0° in this view) to the camera direction
            # Ensure theta1 < theta2 for correct arc direction
            theta_zx_deg = np.degrees(theta_zx)
            theta1, theta2 = (0, theta_zx_deg) if theta_zx_deg > 0 else (theta_zx_deg, 0)
            arc = Arc((E[2], E[0]), 2*arc_radius, 2*arc_radius,
                     angle=0, theta1=theta1, theta2=theta2,
                     color='orange', linewidth=2.5, zorder=5)
            ax.add_patch(arc)
            
            # Label
            mid_angle = theta_zx / 2
            label_r = arc_radius * 1.4
            label_z = E[2] + label_r * np.cos(mid_angle)
            label_x = E[0] + label_r * np.sin(mid_angle)
            ax.text(label_z, label_x, f'θ(x)={theta_zx_deg:.1f}°',
                   fontsize=9, color='orange', fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='orange', alpha=0.9))
        
        ax.set_xlabel('z (mm)', fontsize=10)
        ax.set_ylabel('x (mm)', fontsize=10)
        ax.set_title('Top View (z-x plane)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left', fontsize=8)
        
        # --- Subplot 3: y-z plane (front view from x-axis) ---
        ax = axes[2]
        ax.set_aspect('equal')
        ax.axhline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
        
        # Screen (z-y projection, rotated: z horizontal, y vertical)
        screen_zy = screen_corners[:, [2, 1]]  # Swap to [z, y]
        ax.plot(screen_zy[:, 0], screen_zy[:, 1], 'k-', linewidth=2, label='Screen')
        
        # Eye (swap coordinates: z on x-axis, y on y-axis)
        eye_circle = plt.Circle((E[2], E[1]), eye_radius, color='steelblue',
                                ec='black', linewidth=2, zorder=10)
        ax.add_patch(eye_circle)
        ax.plot(E[2], E[1], 'k.', markersize=6, zorder=11)
        ax.text(E[2] - eye_radius*1.5, E[1], 'E', ha='right', va='center',
                fontsize=10, fontweight='bold')
        
        # Camera (swap coordinates)
        cam_rect = mpatches.FancyBboxPatch(
            (C[2]-cam_size*0.4, C[1]-cam_size*0.6), cam_size*0.8, cam_size*1.2,
            boxstyle="round,pad=0.05",
            ec='black', fc='darkred', linewidth=2, zorder=10, alpha=0.7
        )
        ax.add_patch(cam_rect)
        ax.plot(C[2], C[1], 'wo', markersize=8, zorder=11)
        ax.text(C[2], C[1] - cam_size*0.7, 'C', ha='center', va='top',
                fontsize=10, fontweight='bold', color='white')
        
        # Eye-camera line (swap coordinates)
        ax.plot([E[2], C[2]], [E[1], C[1]], 'r--', alpha=0.5, linewidth=1.5)
        
        # Eye-screen line (swap coordinates)
        ax.plot([E[2], d], [E[1], 0], 'g--', alpha=0.5, linewidth=1.5)
        
        # Theta angle annotation (in z-y plane, showing y-component)
        # This shows the projection of theta onto the z-y plane
        from matplotlib.patches import Arc
        # Calculate the angle in the z-y plane
        theta_zy = np.arctan2(C[1], C[2])  # angle from z-axis in z-y plane
        if abs(theta_zy) > 1e-6:
            arc_radius = min(r, d) * 0.3
            # Arc from z-axis (0° in this view) to the camera direction
            # Ensure theta1 < theta2 for correct arc direction
            theta_zy_deg = np.degrees(theta_zy)
            theta1, theta2 = (0, theta_zy_deg) if theta_zy_deg > 0 else (theta_zy_deg, 0)
            arc = Arc((E[2], E[1]), 2*arc_radius, 2*arc_radius,
                     angle=0, theta1=theta1, theta2=theta2,
                     color='orange', linewidth=2.5, zorder=5)
            ax.add_patch(arc)
            
            # Label
            mid_angle = theta_zy / 2
            label_r = arc_radius * 1.4
            label_z = E[2] + label_r * np.cos(mid_angle)
            label_y = E[1] + label_r * np.sin(mid_angle)
            ax.text(label_z, label_y, f'θ(y)={theta_zy_deg:.1f}°',
                   fontsize=9, color='orange', fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='orange', alpha=0.9))
        
        ax.set_xlabel('z (mm)', fontsize=10)
        ax.set_ylabel('y (mm)', fontsize=10)
        ax.set_title('Side View (z-y plane)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        # Overall title
        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)
        fig.suptitle(f'Eye-Tracking Setup Geometry: θ={theta_deg:.1f}°, φ={phi_deg:.1f}°',
                     fontsize=13, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        return fig, axes