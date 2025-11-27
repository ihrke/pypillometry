"""
ExperimentalSetup: Geometric model of eye-tracking experimental setup.

This module provides a unified representation of the geometric parameters
that define an eye-tracking setup, including screen geometry, eye position,
camera position, and screen orientation.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from ..units import parse_distance, parse_angle
from loguru import logger


class ExperimentalSetup:
    """
    Complete geometric model of an eye-tracking experimental setup.
    
    Encapsulates:
    - Screen geometry (resolution, physical size)
    - Eye position relative to screen (perpendicular distance and offset)
    - Screen orientation (pitch and yaw tilt angles)
    - Camera position (relative to screen or eye)
    - Inter-pupillary distance (for binocular setups)
    
    Provides:
    - Coordinate conversion between pixels and mm
    - Transformation from screen coordinates to eye-centered 3D frame
    - Validation of parameter consistency
    - Visualization of the setup geometry
    - Serialization for saving/loading
    
    Parameters
    ----------
    screen_resolution : tuple
        Screen resolution in pixels (width, height)
    physical_screen_size : tuple
        Physical screen dimensions. Each element can be:
        - float: assumed to be mm
        - str: with units, e.g., "52 cm", "520 mm"
    eye_to_screen_perpendicular : float, str, optional
        Perpendicular distance from eye to screen plane (d).
        This is the cleanest geometric definition.
        - float: assumed to be mm
        - str: with units, e.g., "65 cm", "650 mm"
    eye_offset : tuple, optional
        Eye offset from screen center (delta_x, delta_y).
        Each element can be float (mm) or str with units.
        Positive delta_x = eye right of screen center.
        Positive delta_y = eye above screen center.
    eye_to_screen_center : float, str, optional
        Alternative: direct distance from eye to screen center.
        If provided with eye_offset, d is computed from:
        d = sqrt(distance^2 - delta_x^2 - delta_y^2)
    screen_pitch : float, str, optional
        Screen pitch angle (alpha_tilt). Rotation about horizontal axis.
        Positive = top of screen tilted away from eye.
        - float: assumed to be radians
        - str: with units, e.g., "-5 deg", "0.087 rad"
    screen_yaw : float, str, optional
        Screen yaw angle (beta_tilt). Rotation about vertical axis.
        Positive = right side of screen closer to eye.
        - float: assumed to be radians
        - str: with units, e.g., "3 deg", "0.052 rad"
    camera_position_relative_to : str, optional
        Reference frame for camera position: "screen" or "eye".
        Default is "screen" (camera fixed to screen, moves with tilt).
    camera_offset : tuple, optional
        Camera position as (x, y, z) offset from reference.
        Each element can be float (mm) or str with units.
        For "screen" reference: z=0 is screen plane, negative z is toward eye.
        For "eye" reference: z is depth (toward screen).
    camera_spherical : tuple, optional
        Alternative: camera position in spherical coordinates (theta, phi, r).
        theta: polar angle from z-axis (radians or str)
        phi: azimuthal angle in x-y plane (radians or str)
        r: distance from reference (mm or str)
    ipd : float, str, optional
        Inter-pupillary distance for binocular setups.
        - float: assumed to be mm
        - str: with units, e.g., "63 mm", "6.3 cm"
    
    Attributes
    ----------
    screen_resolution : tuple
        (width, height) in pixels
    physical_screen_size : tuple
        (width, height) in mm
    d : float
        Perpendicular eye-to-screen distance in mm
    delta_x : float
        Eye x-offset from screen center in mm
    delta_y : float
        Eye y-offset from screen center in mm
    alpha_tilt : float
        Screen pitch angle in radians
    beta_tilt : float
        Screen yaw angle in radians
    r : float
        Eye-to-camera distance in mm
    theta : float
        Camera polar angle in radians (eye-centered spherical coords)
    phi : float
        Camera azimuthal angle in radians (eye-centered spherical coords)
    ipd : float or None
        Inter-pupillary distance in mm
    
    Examples
    --------
    >>> # Basic setup with perpendicular distance
    >>> setup = ExperimentalSetup(
    ...     screen_resolution=(1920, 1080),
    ...     physical_screen_size=("52 cm", "29 cm"),
    ...     eye_to_screen_perpendicular="65 cm",
    ...     camera_offset=("0 cm", "-30 cm", "0 cm"),  # camera below screen
    ... )
    >>> print(f"d = {setup.d} mm")
    
    >>> # Setup with eye offset and screen tilt
    >>> setup = ExperimentalSetup(
    ...     screen_resolution=(1920, 1080),
    ...     physical_screen_size=("52 cm", "29 cm"),
    ...     eye_to_screen_perpendicular="65 cm",
    ...     eye_offset=("-3 cm", "1.5 cm"),  # eye left and above center
    ...     screen_pitch="-5 deg",  # screen tilted back
    ...     camera_offset=("0 cm", "-30 cm", "0 cm"),
    ... )
    
    >>> # Visualize the setup
    >>> setup.plot(view='side')
    """
    
    def __init__(
        self,
        screen_resolution: Tuple[int, int],
        physical_screen_size: Tuple[Union[float, str], Union[float, str]],
        eye_to_screen_perpendicular: Optional[Union[float, str]] = None,
        eye_offset: Optional[Tuple[Union[float, str], Union[float, str]]] = None,
        eye_to_screen_center: Optional[Union[float, str]] = None,
        screen_pitch: Union[float, str] = 0.0,
        screen_yaw: Union[float, str] = 0.0,
        camera_position_relative_to: str = "screen",
        camera_offset: Optional[Tuple[Union[float, str], Union[float, str], Union[float, str]]] = None,
        camera_spherical: Optional[Tuple[Union[float, str], Union[float, str], Union[float, str]]] = None,
        ipd: Optional[Union[float, str]] = None,
    ):
        # Screen geometry
        self._screen_resolution = tuple(screen_resolution)
        self._physical_screen_size = self._parse_screen_size(physical_screen_size)
        
        # Eye position
        self._eye_offset = self._parse_eye_offset(eye_offset)
        self._d = self._compute_d(eye_to_screen_perpendicular, eye_to_screen_center)
        
        # Screen orientation
        self._alpha_tilt = parse_angle(screen_pitch) if screen_pitch else 0.0
        self._beta_tilt = parse_angle(screen_yaw) if screen_yaw else 0.0
        
        # Camera position
        self._camera_relative_to = camera_position_relative_to
        self._camera_offset, self._camera_spherical = self._parse_camera_position(
            camera_offset, camera_spherical
        )
        
        # Binocular
        self._ipd = parse_distance(ipd) if ipd is not None else None
        
        # Validate
        self.validate()
    
    # =========================================================================
    # Parsing helpers
    # =========================================================================
    
    def _parse_screen_size(
        self, 
        physical_screen_size: Tuple[Union[float, str], Union[float, str]]
    ) -> Tuple[float, float]:
        """Parse physical screen size to mm."""
        return (
            parse_distance(physical_screen_size[0]),
            parse_distance(physical_screen_size[1])
        )
    
    def _parse_eye_offset(
        self, 
        eye_offset: Optional[Tuple[Union[float, str], Union[float, str]]]
    ) -> Tuple[float, float]:
        """Parse eye offset to mm, defaulting to (0, 0)."""
        if eye_offset is None:
            return (0.0, 0.0)
        return (
            parse_distance(eye_offset[0]),
            parse_distance(eye_offset[1])
        )
    
    def _compute_d(
        self,
        eye_to_screen_perpendicular: Optional[Union[float, str]],
        eye_to_screen_center: Optional[Union[float, str]]
    ) -> Optional[float]:
        """
        Compute perpendicular distance d from available information.
        
        If eye_to_screen_perpendicular is given, use it directly.
        If eye_to_screen_center is given, compute d from:
            d = sqrt(distance^2 - delta_x^2 - delta_y^2)
        """
        if eye_to_screen_perpendicular is not None:
            return parse_distance(eye_to_screen_perpendicular)
        
        if eye_to_screen_center is not None:
            dist_center = parse_distance(eye_to_screen_center)
            delta_x, delta_y = self._eye_offset
            d_squared = dist_center**2 - delta_x**2 - delta_y**2
            if d_squared < 0:
                raise ValueError(
                    f"eye_to_screen_center ({dist_center} mm) is less than "
                    f"eye_offset magnitude ({np.sqrt(delta_x**2 + delta_y**2):.1f} mm). "
                    "This is geometrically impossible."
                )
            return np.sqrt(d_squared)
        
        return None
    
    def _parse_camera_position(
        self,
        camera_offset: Optional[Tuple],
        camera_spherical: Optional[Tuple]
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """Parse camera position to (offset_tuple, spherical_tuple)."""
        parsed_offset = None
        parsed_spherical = None
        
        if camera_offset is not None:
            parsed_offset = (
                parse_distance(camera_offset[0]),
                parse_distance(camera_offset[1]),
                parse_distance(camera_offset[2])
            )
        
        if camera_spherical is not None:
            parsed_spherical = (
                parse_angle(camera_spherical[0]),
                parse_angle(camera_spherical[1]),
                parse_distance(camera_spherical[2])
            )
        
        return parsed_offset, parsed_spherical
    
    # =========================================================================
    # Screen properties
    # =========================================================================
    
    @property
    def screen_resolution(self) -> Tuple[int, int]:
        """Screen resolution (width, height) in pixels."""
        return self._screen_resolution
    
    @property
    def screen_width(self) -> int:
        """Screen width in pixels."""
        return self._screen_resolution[0]
    
    @property
    def screen_height(self) -> int:
        """Screen height in pixels."""
        return self._screen_resolution[1]
    
    @property
    def physical_screen_size(self) -> Tuple[float, float]:
        """Physical screen size (width, height) in mm."""
        return self._physical_screen_size
    
    @property
    def physical_screen_width(self) -> float:
        """Physical screen width in mm."""
        return self._physical_screen_size[0]
    
    @property
    def physical_screen_height(self) -> float:
        """Physical screen height in mm."""
        return self._physical_screen_size[1]
    
    @property
    def screen_xlim(self) -> Tuple[int, int]:
        """Screen x-limits in pixels (0, width)."""
        return (0, self._screen_resolution[0])
    
    @property
    def screen_ylim(self) -> Tuple[int, int]:
        """Screen y-limits in pixels (0, height)."""
        return (0, self._screen_resolution[1])
    
    @property
    def mm_per_pixel_x(self) -> float:
        """Conversion factor: mm per pixel in x direction."""
        return self._physical_screen_size[0] / self._screen_resolution[0]
    
    @property
    def mm_per_pixel_y(self) -> float:
        """Conversion factor: mm per pixel in y direction."""
        return self._physical_screen_size[1] / self._screen_resolution[1]
    
    # =========================================================================
    # Eye position properties
    # =========================================================================
    
    @property
    def d(self) -> float:
        """Perpendicular eye-to-screen distance in mm."""
        if self._d is None:
            raise ValueError(
                "Eye-to-screen distance not set. Provide either "
                "eye_to_screen_perpendicular or eye_to_screen_center."
            )
        return self._d
    
    @property
    def delta_x(self) -> float:
        """Eye x-offset from screen center in mm (positive = eye right of center)."""
        return self._eye_offset[0]
    
    @property
    def delta_y(self) -> float:
        """Eye y-offset from screen center in mm (positive = eye above center)."""
        return self._eye_offset[1]
    
    @property
    def eye_offset(self) -> Tuple[float, float]:
        """Eye offset from screen center (delta_x, delta_y) in mm."""
        return self._eye_offset
    
    @property
    def eye_to_screen_center(self) -> float:
        """Direct distance from eye to screen center in mm."""
        return np.sqrt(self.d**2 + self.delta_x**2 + self.delta_y**2)
    
    # =========================================================================
    # Screen orientation properties
    # =========================================================================
    
    @property
    def alpha_tilt(self) -> float:
        """Screen pitch angle in radians (positive = top tilted away)."""
        return self._alpha_tilt
    
    @property
    def beta_tilt(self) -> float:
        """Screen yaw angle in radians (positive = right side closer)."""
        return self._beta_tilt
    
    @property
    def screen_pitch(self) -> float:
        """Alias for alpha_tilt (screen pitch in radians)."""
        return self._alpha_tilt
    
    @property
    def screen_yaw(self) -> float:
        """Alias for beta_tilt (screen yaw in radians)."""
        return self._beta_tilt
    
    # =========================================================================
    # Camera position properties
    # =========================================================================
    
    @property
    def camera_position_relative_to(self) -> str:
        """Reference frame for camera position: 'screen' or 'eye'."""
        return self._camera_relative_to
    
    @property
    def camera_offset_screen_frame(self) -> Optional[Tuple[float, float, float]]:
        """Camera position (x, y, z) in screen frame in mm, or None if not set."""
        if self._camera_offset is not None:
            return self._camera_offset
        if self._camera_spherical is not None:
            # Convert spherical to Cartesian in screen frame
            theta, phi, r = self._camera_spherical
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return (x, y, z)
        return None
    
    @property
    def r(self) -> float:
        """Eye-to-camera distance in mm."""
        # First, get camera position in eye frame
        cam_eye = self.camera_position_eye_frame
        if cam_eye is None:
            raise ValueError("Camera position not set.")
        return np.sqrt(cam_eye[0]**2 + cam_eye[1]**2 + cam_eye[2]**2)
    
    @property
    def theta(self) -> float:
        """Camera polar angle from z-axis in eye-centered coords (radians)."""
        cam_eye = self.camera_position_eye_frame
        if cam_eye is None:
            raise ValueError("Camera position not set.")
        r = self.r
        return np.arccos(cam_eye[2] / r)
    
    @property
    def phi(self) -> float:
        """Camera azimuthal angle in eye-centered coords (radians)."""
        cam_eye = self.camera_position_eye_frame
        if cam_eye is None:
            raise ValueError("Camera position not set.")
        return np.arctan2(cam_eye[1], cam_eye[0])
    
    @property
    def camera_position_eye_frame(self) -> Optional[Tuple[float, float, float]]:
        """
        Camera position (cx, cy, cz) in eye-centered frame in mm.
        
        Eye is at origin, z-axis points toward screen center (before tilt).
        """
        if self._camera_offset is None and self._camera_spherical is None:
            return None
        
        # Get camera position in screen frame
        cam_screen = self.camera_offset_screen_frame
        
        if self._camera_relative_to == "eye":
            # Already in eye frame
            return cam_screen
        
        # Camera is in screen frame - need to transform to eye frame
        # Screen center in eye frame (before tilt) is at (delta_x, delta_y, d)
        # Camera position in screen frame: (cx_s, cy_s, cz_s) where cz_s is 
        # typically 0 or small (camera at/near screen plane)
        cx_s, cy_s, cz_s = cam_screen
        
        # Apply screen tilt rotation
        # The screen is rotated, so camera rotates with it
        cos_a, sin_a = np.cos(self._alpha_tilt), np.sin(self._alpha_tilt)
        cos_b, sin_b = np.cos(self._beta_tilt), np.sin(self._beta_tilt)
        
        # Rotation matrix R = R_yaw @ R_pitch (same as in algorithm doc)
        # Transform camera offset from screen frame to eye frame
        cx_rot = cx_s * cos_b + cy_s * sin_a * sin_b - cz_s * cos_a * sin_b
        cy_rot = cy_s * cos_a + cz_s * sin_a
        cz_rot = cx_s * sin_b - cy_s * sin_a * cos_b + cz_s * cos_a * cos_b
        
        # Add screen center position (in eye frame)
        cx_eye = cx_rot + self.delta_x
        cy_eye = cy_rot + self.delta_y
        cz_eye = cz_rot + self.d
        
        return (cx_eye, cy_eye, cz_eye)
    
    # =========================================================================
    # Binocular properties
    # =========================================================================
    
    @property
    def ipd(self) -> Optional[float]:
        """Inter-pupillary distance in mm, or None if not set."""
        return self._ipd
    
    @property
    def left_eye_offset(self) -> float:
        """X-offset for left eye in binocular setup (mm)."""
        if self._ipd is None:
            return 0.0
        return -self._ipd / 2
    
    @property
    def right_eye_offset(self) -> float:
        """X-offset for right eye in binocular setup (mm)."""
        if self._ipd is None:
            return 0.0
        return self._ipd / 2
    
    # =========================================================================
    # Coordinate conversion methods
    # =========================================================================
    
    def pixels_to_mm(
        self, 
        x_px: Union[float, np.ndarray], 
        y_px: Union[float, np.ndarray],
        centered: bool = True
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert pixel coordinates to mm.
        
        Parameters
        ----------
        x_px, y_px : float or array
            Coordinates in pixels
        centered : bool, default True
            If True, output is relative to screen center.
            If False, output is relative to screen origin (0,0).
        
        Returns
        -------
        x_mm, y_mm : float or array
            Coordinates in mm
        """
        x_mm = x_px * self.mm_per_pixel_x
        y_mm = y_px * self.mm_per_pixel_y
        
        if centered:
            x_mm = x_mm - self._physical_screen_size[0] / 2
            y_mm = y_mm - self._physical_screen_size[1] / 2
        
        return x_mm, y_mm
    
    def mm_to_pixels(
        self, 
        x_mm: Union[float, np.ndarray], 
        y_mm: Union[float, np.ndarray],
        centered: bool = True
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert mm coordinates to pixels.
        
        Parameters
        ----------
        x_mm, y_mm : float or array
            Coordinates in mm
        centered : bool, default True
            If True, input is relative to screen center.
            If False, input is relative to screen origin (0,0).
        
        Returns
        -------
        x_px, y_px : float or array
            Coordinates in pixels
        """
        if centered:
            x_mm = x_mm + self._physical_screen_size[0] / 2
            y_mm = y_mm + self._physical_screen_size[1] / 2
        
        x_px = x_mm / self.mm_per_pixel_x
        y_px = y_mm / self.mm_per_pixel_y
        
        return x_px, y_px
    
    def screen_point_to_eye_frame(
        self,
        x_mm: Union[float, np.ndarray],
        y_mm: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Transform screen point(s) from screen-centered mm to eye-centered 3D frame.
        
        Applies screen tilt rotation and eye offset translation.
        
        Parameters
        ----------
        x_mm, y_mm : float or array
            Screen coordinates in mm (relative to screen center)
        
        Returns
        -------
        T_x, T_y, T_z : float or array
            3D coordinates in eye-centered frame (mm)
        
        Notes
        -----
        Uses the transformation from algorithm document Section 10.7.3:
        T(x_s, y_s) = R(alpha, beta) @ [x_s, y_s, 0]^T + [delta_x, delta_y, d]^T
        """
        cos_a, sin_a = np.cos(self._alpha_tilt), np.sin(self._alpha_tilt)
        cos_b, sin_b = np.cos(self._beta_tilt), np.sin(self._beta_tilt)
        
        # Apply rotation and translation
        T_x = x_mm * cos_b + y_mm * sin_a * sin_b + self.delta_x
        T_y = y_mm * cos_a + self.delta_y
        T_z = -x_mm * sin_b + y_mm * sin_a * cos_b + self.d * cos_a * cos_b
        
        return T_x, T_y, T_z
    
    # =========================================================================
    # Validation
    # =========================================================================
    
    def validate(self) -> None:
        """
        Validate parameter consistency. Raises ValueError if invalid.
        """
        # Screen resolution must be positive
        if self._screen_resolution[0] <= 0 or self._screen_resolution[1] <= 0:
            raise ValueError(
                f"Screen resolution must be positive, got {self._screen_resolution}"
            )
        
        # Physical screen size must be positive
        if self._physical_screen_size[0] <= 0 or self._physical_screen_size[1] <= 0:
            raise ValueError(
                f"Physical screen size must be positive, got {self._physical_screen_size}"
            )
        
        # Tilt angles should be reasonable (< 45 degrees)
        if abs(self._alpha_tilt) > np.pi/4:
            logger.warning(
                f"Screen pitch {np.degrees(self._alpha_tilt):.1f}° is large (>45°). "
                "Check if this is intended."
            )
        if abs(self._beta_tilt) > np.pi/4:
            logger.warning(
                f"Screen yaw {np.degrees(self._beta_tilt):.1f}° is large (>45°). "
                "Check if this is intended."
            )
        
        # Camera reference frame must be valid
        if self._camera_relative_to not in ("screen", "eye"):
            raise ValueError(
                f"camera_position_relative_to must be 'screen' or 'eye', "
                f"got '{self._camera_relative_to}'"
            )
    
    def is_complete(self) -> bool:
        """
        Check if setup has all parameters needed for foreshortening correction.
        
        Returns
        -------
        bool
            True if d and camera position are set.
        """
        return (
            self._d is not None and
            (self._camera_offset is not None or self._camera_spherical is not None)
        )
    
    def has_screen_info(self) -> bool:
        """Check if screen resolution and physical size are set."""
        return (
            self._screen_resolution is not None and
            self._physical_screen_size is not None
        )
    
    def has_eye_distance(self) -> bool:
        """Check if eye-to-screen distance is set."""
        return self._d is not None
    
    def has_camera_position(self) -> bool:
        """Check if camera position is set."""
        return self._camera_offset is not None or self._camera_spherical is not None
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export setup to dictionary for JSON serialization.
        
        Returns
        -------
        dict
            All setup parameters as a dictionary
        """
        return {
            'screen_resolution': self._screen_resolution,
            'physical_screen_size': self._physical_screen_size,
            'd': self._d,
            'eye_offset': self._eye_offset,
            'alpha_tilt': self._alpha_tilt,
            'beta_tilt': self._beta_tilt,
            'camera_position_relative_to': self._camera_relative_to,
            'camera_offset': self._camera_offset,
            'camera_spherical': self._camera_spherical,
            'ipd': self._ipd,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentalSetup':
        """
        Create ExperimentalSetup from dictionary.
        
        Parameters
        ----------
        d : dict
            Dictionary from to_dict()
        
        Returns
        -------
        ExperimentalSetup
        """
        return cls(
            screen_resolution=d['screen_resolution'],
            physical_screen_size=d['physical_screen_size'],
            eye_to_screen_perpendicular=d.get('d'),
            eye_offset=d.get('eye_offset'),
            screen_pitch=d.get('alpha_tilt', 0.0),
            screen_yaw=d.get('beta_tilt', 0.0),
            camera_position_relative_to=d.get('camera_position_relative_to', 'screen'),
            camera_offset=d.get('camera_offset'),
            camera_spherical=d.get('camera_spherical'),
            ipd=d.get('ipd'),
        )
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def plot(
        self, 
        ax=None, 
        view: str = 'top',
        show_camera: bool = True,
        show_eye: bool = True,
        show_screen: bool = True,
        figsize: Tuple[float, float] = (8, 6)
    ):
        """
        Visualize the experimental setup geometry.
        
        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates new figure.
        view : str, default 'top'
            View perspective: 'top' (x-z plane), 'side' (y-z plane), or '3d'
        show_camera : bool, default True
            Whether to show camera position
        show_eye : bool, default True
            Whether to show eye position
        show_screen : bool, default True
            Whether to show screen
        figsize : tuple, default (8, 6)
            Figure size if creating new figure
        
        Returns
        -------
        ax : matplotlib Axes
            The axes with the plot
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if ax is None:
            fig = plt.figure(figsize=figsize)
            if view == '3d':
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        
        # Screen corners in screen frame (mm, centered)
        w, h = self._physical_screen_size
        screen_corners = [
            (-w/2, -h/2),  # bottom-left
            (w/2, -h/2),   # bottom-right
            (w/2, h/2),    # top-right
            (-w/2, h/2),   # top-left
            (-w/2, -h/2),  # close polygon
        ]
        
        # Transform screen corners to eye frame
        screen_3d = []
        for x, y in screen_corners:
            T = self.screen_point_to_eye_frame(x, y)
            screen_3d.append(T)
        
        # Eye position (at origin in eye frame)
        eye_pos = (0, 0, 0)
        
        # Camera position
        cam_pos = self.camera_position_eye_frame if self.has_camera_position() else None
        
        if view == 'top':
            # x-z plane (looking down from above)
            if show_screen:
                xs = [p[0] for p in screen_3d]
                zs = [p[2] for p in screen_3d]
                ax.plot(xs, zs, 'b-', linewidth=2, label='Screen')
                ax.fill(xs, zs, alpha=0.2, color='blue')
            
            if show_eye:
                ax.plot(eye_pos[0], eye_pos[2], 'go', markersize=10, label='Eye')
            
            if show_camera and cam_pos is not None:
                ax.plot(cam_pos[0], cam_pos[2], 'r^', markersize=10, label='Camera')
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Z (mm) - depth')
            ax.set_title('Top View (X-Z plane)')
            
        elif view == 'side':
            # y-z plane (looking from the side)
            if show_screen:
                ys = [p[1] for p in screen_3d]
                zs = [p[2] for p in screen_3d]
                ax.plot(ys, zs, 'b-', linewidth=2, label='Screen')
                ax.fill(ys, zs, alpha=0.2, color='blue')
            
            if show_eye:
                ax.plot(eye_pos[1], eye_pos[2], 'go', markersize=10, label='Eye')
            
            if show_camera and cam_pos is not None:
                ax.plot(cam_pos[1], cam_pos[2], 'r^', markersize=10, label='Camera')
            
            ax.set_xlabel('Y (mm)')
            ax.set_ylabel('Z (mm) - depth')
            ax.set_title('Side View (Y-Z plane)')
            
        elif view == '3d':
            if show_screen:
                xs = [p[0] for p in screen_3d]
                ys = [p[1] for p in screen_3d]
                zs = [p[2] for p in screen_3d]
                ax.plot(xs, ys, zs, 'b-', linewidth=2, label='Screen')
            
            if show_eye:
                ax.scatter([eye_pos[0]], [eye_pos[1]], [eye_pos[2]], 
                          c='green', s=100, marker='o', label='Eye')
            
            if show_camera and cam_pos is not None:
                ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], 
                          c='red', s=100, marker='^', label='Camera')
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('3D View')
        
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    # =========================================================================
    # String representation
    # =========================================================================
    
    def __repr__(self) -> str:
        """Concise representation."""
        parts = [f"ExperimentalSetup("]
        parts.append(f"  screen={self._screen_resolution[0]}x{self._screen_resolution[1]}px")
        parts.append(f"  physical={self._physical_screen_size[0]:.0f}x{self._physical_screen_size[1]:.0f}mm")
        
        if self._d is not None:
            parts.append(f"  d={self._d:.0f}mm")
        
        if self._eye_offset != (0.0, 0.0):
            parts.append(f"  eye_offset=({self._eye_offset[0]:.1f}, {self._eye_offset[1]:.1f})mm")
        
        if self._alpha_tilt != 0.0 or self._beta_tilt != 0.0:
            parts.append(f"  tilt=({np.degrees(self._alpha_tilt):.1f}°, {np.degrees(self._beta_tilt):.1f}°)")
        
        if self.has_camera_position():
            parts.append(f"  camera: r={self.r:.0f}mm, θ={np.degrees(self.theta):.1f}°, φ={np.degrees(self.phi):.1f}°")
        
        if self._ipd is not None:
            parts.append(f"  ipd={self._ipd:.0f}mm")
        
        parts.append(")")
        return "\n".join(parts)
    
    def __str__(self) -> str:
        """Human-readable string."""
        return self.__repr__()

