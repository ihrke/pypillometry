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
    screen_offset : tuple, optional
        Offset of screen center from the eye reference point (delta_x, delta_y).
        Each element can be float (mm) or str with units.
        Positive delta_x = screen center is to the right of eye.
        Positive delta_y = screen center is above eye.
        Note: For binocular setups, the reference is the midpoint between
        the two eyes (root of nose).
    eye_to_screen_center : float, str, optional
        Distance from eye to screen center.
        - float: assumed to be mm
        - str: with units, e.g., "65 cm", "650 mm"
        The perpendicular distance d between midpoint and eye-plane is computed from:
        d = sqrt(eye_to_screen_center^2 - delta_x^2 - delta_y^2)
        If screen_offset is (0, 0) or not set, d equals eye_to_screen_center.
        Note: For binocular setups, "eye" refers to the midpoint between
        the two eyes (root of nose).
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
    camera_offset : tuple, optional
        Camera position as (x, y, z) offset from eye in mm.
        Each element can be float (mm) or str with units.
        z-axis points toward screen center.
    camera_spherical : tuple, optional
        Alternative: camera position in spherical coordinates (theta, phi, r)
        relative to eye.
        theta: polar angle from z-axis (radians or str)
        phi: azimuthal angle in x-y plane (radians or str)
        r: distance from eye (mm or str)
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
        Screen center x-offset from eye in mm (positive = right)
    delta_y : float
        Screen center y-offset from eye in mm (positive = up)
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
    
    Notes
    -----
    All coordinates are in an eye-centric reference frame where:
    - The origin is at the eye position
    - For binocular setups, this is the midpoint between the two eyes (root of nose)
    - z-axis points toward the screen center
    - x-axis points to the right
    - y-axis points upward
    
    Examples
    --------
    >>> # Basic setup with distance to screen center
    >>> setup = ExperimentalSetup(
    ...     screen_resolution=(1920, 1080),
    ...     physical_screen_size=("52 cm", "29 cm"),
    ...     eye_to_screen_center="65 cm",
    ...     camera_offset=("0 cm", "-30 cm", "0 cm"),  # camera below screen
    ... )
    >>> print(f"d = {setup.d} mm")
    
    >>> # Setup with screen offset and tilt
    >>> setup = ExperimentalSetup(
    ...     screen_resolution=(1920, 1080),
    ...     physical_screen_size=("52 cm", "29 cm"),
    ...     eye_to_screen_center="65 cm",
    ...     screen_offset=("3 cm", "-1.5 cm"),  # screen right and below eye
    ...     screen_pitch="-5 deg",  # screen tilted back
    ...     camera_offset=("0 cm", "-30 cm", "0 cm"),
    ... )
    
    >>> # Visualize the setup
    >>> setup.plot(view='side')
    """
    
    def __init__(
        self,
        screen_resolution: Optional[Tuple[int, int]] = None,
        physical_screen_size: Optional[Tuple[Union[float, str], Union[float, str]]] = None,
        screen_offset: Optional[Tuple[Union[float, str], Union[float, str]]] = None,
        eye_to_screen_center: Optional[Union[float, str]] = None,
        screen_pitch: Optional[Union[float, str]] = None,
        screen_yaw: Optional[Union[float, str]] = None,
        camera_offset: Optional[Tuple[Union[float, str], Union[float, str], Union[float, str]]] = None,
        camera_spherical: Optional[Tuple[Union[float, str], Union[float, str], Union[float, str]]] = None,
        ipd: Optional[Union[float, str]] = None,
    ):
        # Screen geometry (optional)
        self._screen_resolution = tuple(screen_resolution) if screen_resolution is not None else None
        self._physical_screen_size = self._parse_screen_size(physical_screen_size) if physical_screen_size is not None else None
        
        # Screen offset from eye (eye-centric coordinate system)
        self._screen_offset = self._parse_screen_offset(screen_offset)
        self._d = self._compute_d(eye_to_screen_center)
        
        # Screen orientation
        self._alpha_tilt = parse_angle(screen_pitch) if screen_pitch else 0.0
        self._beta_tilt = parse_angle(screen_yaw) if screen_yaw else 0.0
        
        # Camera position (always in eye-centric frame)
        self._camera_offset, self._camera_spherical = self._parse_camera_position(
            camera_offset, camera_spherical
        )
        
        # Binocular
        self._ipd = parse_distance(ipd) if ipd is not None else None
        
        # Validate (only if there's something to validate)
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
    
    def _parse_screen_offset(
        self, 
        screen_offset: Optional[Tuple[Union[float, str], Union[float, str]]]
    ) -> Tuple[float, float]:
        """Parse screen offset to mm, defaulting to (0, 0)."""
        if screen_offset is None:
            return (0.0, 0.0)
        return (
            parse_distance(screen_offset[0]),
            parse_distance(screen_offset[1])
        )
    
    def _compute_d(
        self,
        eye_to_screen_center: Optional[Union[float, str]]
    ) -> Optional[float]:
        """
        Compute perpendicular distance d from eye_to_screen_center.
        
        d = sqrt(eye_to_screen_center^2 - delta_x^2 - delta_y^2)
        
        If screen_offset is (0, 0), d equals eye_to_screen_center.
        """
        if eye_to_screen_center is not None:
            dist_center = parse_distance(eye_to_screen_center)
            delta_x, delta_y = self._screen_offset
            d_squared = dist_center**2 - delta_x**2 - delta_y**2
            if d_squared < 0:
                raise ValueError(
                    f"eye_to_screen_center ({dist_center} mm) is less than "
                    f"screen_offset magnitude ({np.sqrt(delta_x**2 + delta_y**2):.1f} mm). "
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
    
    def _require_screen_resolution(self):
        """Raise error if screen resolution is not set."""
        if self._screen_resolution is None:
            raise ValueError("Screen resolution not set.")
    
    def _require_physical_screen_size(self):
        """Raise error if physical screen size is not set."""
        if self._physical_screen_size is None:
            raise ValueError("Physical screen size not set.")
    
    @property
    def screen_resolution(self) -> Optional[Tuple[int, int]]:
        """Screen resolution (width, height) in pixels, or None if not set."""
        return self._screen_resolution
    
    @property
    def screen_width(self) -> int:
        """Screen width in pixels."""
        self._require_screen_resolution()
        return self._screen_resolution[0]
    
    @property
    def screen_height(self) -> int:
        """Screen height in pixels."""
        self._require_screen_resolution()
        return self._screen_resolution[1]
    
    @property
    def physical_screen_size(self) -> Optional[Tuple[float, float]]:
        """Physical screen size (width, height) in mm, or None if not set."""
        return self._physical_screen_size
    
    @property
    def physical_screen_width(self) -> float:
        """Physical screen width in mm."""
        self._require_physical_screen_size()
        return self._physical_screen_size[0]
    
    @property
    def physical_screen_height(self) -> float:
        """Physical screen height in mm."""
        self._require_physical_screen_size()
        return self._physical_screen_size[1]
    
    @property
    def screen_xlim(self) -> Tuple[int, int]:
        """Screen x-limits in pixels (0, width)."""
        self._require_screen_resolution()
        return (0, self._screen_resolution[0])
    
    @property
    def screen_ylim(self) -> Tuple[int, int]:
        """Screen y-limits in pixels (0, height)."""
        self._require_screen_resolution()
        return (0, self._screen_resolution[1])
    
    @property
    def mm_per_pixel_x(self) -> float:
        """Conversion factor: mm per pixel in x direction."""
        self._require_screen_resolution()
        self._require_physical_screen_size()
        return self._physical_screen_size[0] / self._screen_resolution[0]
    
    @property
    def mm_per_pixel_y(self) -> float:
        """Conversion factor: mm per pixel in y direction."""
        self._require_screen_resolution()
        self._require_physical_screen_size()
        return self._physical_screen_size[1] / self._screen_resolution[1]
    
    # =========================================================================
    # Eye position properties
    # =========================================================================
    
    @property
    def d(self) -> float:
        """Perpendicular eye-to-screen distance in mm."""
        if self._d is None:
            raise ValueError(
                "Eye-to-screen distance not set. Provide eye_to_screen_center."
            )
        return self._d
    
    @property
    def delta_x(self) -> float:
        """Screen center x-offset from eye in mm (positive = screen right of eye)."""
        return self._screen_offset[0]
    
    @property
    def delta_y(self) -> float:
        """Screen center y-offset from eye in mm (positive = screen above eye)."""
        return self._screen_offset[1]
    
    @property
    def screen_offset(self) -> Tuple[float, float]:
        """Screen center offset from eye (delta_x, delta_y) in mm."""
        return self._screen_offset
    
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
    # Camera position properties (always in eye-centric frame)
    # =========================================================================
    
    @property
    def camera_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Camera position (cx, cy, cz) in eye-centered frame in mm.
        
        Eye is at origin, z-axis points toward screen center.
        Returns Cartesian coordinates computed from either camera_offset or camera_spherical.
        """
        if self._camera_offset is not None:
            return self._camera_offset
        if self._camera_spherical is not None:
            # Convert spherical to Cartesian
            theta, phi, r = self._camera_spherical
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return (x, y, z)
        return None
    
    @property
    def r(self) -> float:
        """Eye-to-camera distance in mm."""
        cam = self.camera_position
        if cam is None:
            raise ValueError("Camera position not set.")
        return np.sqrt(cam[0]**2 + cam[1]**2 + cam[2]**2)
    
    @property
    def theta(self) -> float:
        """Camera polar angle from z-axis in eye-centered coords (radians)."""
        cam = self.camera_position
        if cam is None:
            raise ValueError("Camera position not set.")
        r = self.r
        return np.arccos(cam[2] / r)
    
    @property
    def phi(self) -> float:
        """Camera azimuthal angle in eye-centered coords (radians)."""
        cam = self.camera_position
        if cam is None:
            raise ValueError("Camera position not set.")
        return np.arctan2(cam[1], cam[0])
    
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
    
    def mm_to_degrees(
        self,
        x_mm: Union[float, np.ndarray],
        y_mm: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert mm coordinates (relative to screen center) to visual angles.
        
        Parameters
        ----------
        x_mm, y_mm : float or array
            Coordinates in mm relative to screen center.
            Positive x = right, positive y = up.
        
        Returns
        -------
        angle_x, angle_y : float or array
            Visual angles in degrees.
            Positive x = right, positive y = up.
        
        Notes
        -----
        Uses the small-angle approximation: angle = arctan(distance / d)
        where d is the eye-to-screen distance.
        """
        if self._d is None:
            raise ValueError(
                "Eye-to-screen distance not set. Provide eye_to_screen_center."
            )
        angle_x = np.degrees(np.arctan(x_mm / self._d))
        angle_y = np.degrees(np.arctan(y_mm / self._d))
        return angle_x, angle_y
    
    def degrees_to_mm(
        self,
        angle_x: Union[float, str, np.ndarray],
        angle_y: Union[float, str, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert visual angles to mm coordinates (relative to screen center).
        
        Parameters
        ----------
        angle_x, angle_y : float, str, or array
            Visual angles. Can be:
            - float: assumed to be radians
            - str: with units, e.g., "5 deg", "0.087 rad"
            - pint Quantity
            - array of floats (radians)
            Positive x = right, positive y = up.
        
        Returns
        -------
        x_mm, y_mm : float or array
            Coordinates in mm relative to screen center.
            Positive x = right, positive y = up.
        
        Notes
        -----
        Inverse of mm_to_degrees: x_mm = d * tan(angle)
        """
        if self._d is None:
            raise ValueError(
                "Eye-to-screen distance not set. Provide eye_to_screen_center."
            )
        # Parse angles if they're strings or pint quantities
        if isinstance(angle_x, (str,)) or hasattr(angle_x, 'magnitude'):
            angle_x = parse_angle(angle_x)  # returns radians
        else:
            angle_x = parse_angle(angle_x)  # handles float (assumes radians)
        
        if isinstance(angle_y, (str,)) or hasattr(angle_y, 'magnitude'):
            angle_y = parse_angle(angle_y)  # returns radians
        else:
            angle_y = parse_angle(angle_y)  # handles float (assumes radians)
        
        x_mm = self._d * np.tan(angle_x)
        y_mm = self._d * np.tan(angle_y)
        return x_mm, y_mm
    
    def pixels_to_degrees(
        self,
        x_px: Union[float, np.ndarray],
        y_px: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert pixel coordinates to visual angles.
        
        Parameters
        ----------
        x_px, y_px : float or array
            Coordinates in pixels (origin at upper-left, y increases downward).
        
        Returns
        -------
        angle_x, angle_y : float or array
            Visual angles in degrees relative to screen center.
            Positive x = right, positive y = up.
        
        Notes
        -----
        Combines pixels_to_mm (centered) and mm_to_degrees.
        The y-axis is flipped: pixel y increases downward, but visual angle
        y increases upward.
        """
        x_mm, y_mm = self.pixels_to_mm(x_px, y_px, centered=True)
        # Flip y because pixels have +y down, but degrees have +y up
        return self.mm_to_degrees(x_mm, -y_mm)
    
    def degrees_to_pixels(
        self,
        angle_x: Union[float, str, np.ndarray],
        angle_y: Union[float, str, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert visual angles to pixel coordinates.
        
        Parameters
        ----------
        angle_x, angle_y : float, str, or array
            Visual angles. Can be:
            - float: assumed to be radians
            - str: with units, e.g., "5 deg", "0.087 rad"
            - pint Quantity
            - array of floats (radians)
            Positive x = right, positive y = up.
        
        Returns
        -------
        x_px, y_px : float or array
            Coordinates in pixels (origin at upper-left, y increases downward).
        
        Notes
        -----
        Combines degrees_to_mm and mm_to_pixels (centered).
        The y-axis is flipped: visual angle y increases upward, but pixel
        y increases downward.
        """
        x_mm, y_mm = self.degrees_to_mm(angle_x, angle_y)
        # Flip y because degrees have +y up, but pixels have +y down
        return self.mm_to_pixels(x_mm, -y_mm, centered=True)
    
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
        
        Only validates parameters that are set; incomplete setups are allowed.
        """
        # Screen resolution must be positive (if set)
        if self._screen_resolution is not None:
            if self._screen_resolution[0] <= 0 or self._screen_resolution[1] <= 0:
                raise ValueError(
                    f"Screen resolution must be positive, got {self._screen_resolution}"
                )
        
        # Physical screen size must be positive (if set)
        if self._physical_screen_size is not None:
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
    
    def summary(self) -> Dict[str, Any]:
        """
        Return a summary of the experimental setup as a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing setup parameters in human-readable form.
        """
        summary = {
            'screen_resolution': self._screen_resolution,
            'physical_screen_size_mm': self._physical_screen_size,
        }
        
        # Eye position
        if self._d is not None:
            summary['eye_to_screen_distance_mm'] = self._d
        else:
            summary['eye_to_screen_distance_mm'] = "not set"
        
        if self._screen_offset != (0.0, 0.0):
            summary['screen_offset_mm'] = self._screen_offset
        
        # Screen tilt
        if self._alpha_tilt != 0.0 or self._beta_tilt != 0.0:
            summary['screen_tilt_deg'] = (
                np.degrees(self._alpha_tilt),
                np.degrees(self._beta_tilt)
            )
        
        # Camera
        if self.has_camera_position():
            summary['camera_distance_mm'] = self.r
            summary['camera_angles_deg'] = (
                np.degrees(self.theta),
                np.degrees(self.phi)
            )
        else:
            summary['camera_position'] = "not set"
        
        # IPD
        if self._ipd is not None:
            summary['ipd_mm'] = self._ipd
        
        return summary
    
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
            'screen_offset': self._screen_offset,
            'alpha_tilt': self._alpha_tilt,
            'beta_tilt': self._beta_tilt,
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
        # Compute eye_to_screen_center from d and screen_offset
        d_val = d.get('d')
        screen_offset = d.get('screen_offset')
        eye_to_screen_center = None
        if d_val is not None:
            if screen_offset is not None:
                delta_x, delta_y = screen_offset
                eye_to_screen_center = np.sqrt(d_val**2 + delta_x**2 + delta_y**2)
            else:
                eye_to_screen_center = d_val
        
        return cls(
            screen_resolution=d['screen_resolution'],
            physical_screen_size=d['physical_screen_size'],
            screen_offset=screen_offset,
            eye_to_screen_center=eye_to_screen_center,
            screen_pitch=d.get('alpha_tilt', 0.0),
            screen_yaw=d.get('beta_tilt', 0.0),
            camera_offset=d.get('camera_offset'),
            camera_spherical=d.get('camera_spherical'),
            ipd=d.get('ipd'),
        )
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def plot(
        self,
        theta: float = None,
        phi: float = None,
        projection: str = '2d',
        show_gaze_samples: bool = True,
        n_gaze_samples: int = 9,
        ax = None,
        viewing_angle: tuple = None,
    ):
        """
        Visualize the experimental setup geometry.
        
        Parameters
        ----------
        theta : float, optional
            Camera polar angle in radians. If None, uses setup's theta.
        phi : float, optional
            Camera azimuthal angle in radians. If None, uses setup's phi.
        projection : str, default '2d'
            Type of projection:
            - '2d': Three orthogonal 2D views (front, top, side)
            - '3d': Interactive 3D view
        show_gaze_samples : bool, default True
            Whether to show sample gaze vectors to screen
        n_gaze_samples : int, default 9
            Number of sample gaze positions (grid)
        ax : matplotlib Axes, optional
            Existing axis to plot on (only for '3d' projection).
        viewing_angle : tuple, optional
            Viewing angle for 3D projection (elev, azim) or (elev, azim, roll) in degrees.
            Default is (-49, 50, 45) for 3D.
        
        Returns
        -------
        fig : matplotlib Figure
        ax : matplotlib Axes or array of Axes
        
        Examples
        --------
        >>> setup = ExperimentalSetup(
        ...     screen_resolution=(1920, 1080),
        ...     physical_screen_size=("52 cm", "29 cm"),
        ...     eye_to_screen_center="60 cm",
        ...     camera_spherical=("20 deg", "-90 deg", "70 cm"),
        ... )
        >>> fig, ax = setup.plot()  # 2D views
        >>> fig, ax = setup.plot(projection='3d')  # 3D view
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Arc
        
        # Validate required parameters
        if not self.has_screen_info():
            raise ValueError("Cannot plot: physical_screen_size not set")
        if self._d is None:
            raise ValueError("Cannot plot: eye-to-screen distance (d) not set")
        
        # Get camera angles
        if theta is None:
            if not self.has_camera_position():
                raise ValueError("theta not provided and camera position not set")
            theta = self.theta
        if phi is None:
            if not self.has_camera_position():
                raise ValueError("phi not provided and camera position not set")
            phi = self.phi
        
        # Get distances
        r = self.r if self.has_camera_position() else 0
        d = self.d
        w, h = self.physical_screen_width, self.physical_screen_height
        
        if projection.lower() == '3d':
            return self._plot_3d(theta, phi, r, d, w, h, show_gaze_samples, 
                                n_gaze_samples, ax, viewing_angle)
        elif projection.lower() == '2d':
            return self._plot_2d(theta, phi, r, d, w, h, show_gaze_samples, n_gaze_samples)
        else:
            raise ValueError(f"projection must be '2d' or '3d', got '{projection}'")
    
    def _plot_3d(self, theta, phi, r, d, w, h, show_gaze_samples, n_gaze_samples, ax, viewing_angle):
        """3D visualization of experimental setup."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        if viewing_angle is None:
            viewing_angle = (-49, 50, 45)
        
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        eye_pos = np.array([0, 0, 0])
        camera_pos = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])
        screen_corners = np.array([
            [-w/2, -h/2, d], [w/2, -h/2, d], [w/2, h/2, d], [-w/2, h/2, d], [-w/2, -h/2, d],
        ])
        
        ax.scatter(*eye_pos, color='blue', s=200, marker='o', label='Eye', edgecolors='darkblue', linewidths=2, alpha=0.8)
        ax.scatter(*camera_pos, color='red', s=200, marker='^', label='Camera', edgecolors='darkred', linewidths=2, alpha=0.8)
        ax.plot(screen_corners[:, 0], screen_corners[:, 1], screen_corners[:, 2], 'k-', linewidth=2, label='Screen')
        screen_surface = [[screen_corners[0], screen_corners[1], screen_corners[2], screen_corners[3]]]
        ax.add_collection3d(Poly3DCollection(screen_surface, alpha=0.2, facecolor='gray', edgecolor='black'))
        
        ax.plot([0, camera_pos[0]], [0, camera_pos[1]], [0, camera_pos[2]], 'r--', linewidth=2, alpha=0.6, label=f'E-C: {r:.0f}mm')
        ax.plot([0, 0], [0, 0], [0, d], 'g--', linewidth=2, alpha=0.6, label=f'E-S: {d:.0f}mm')
        
        # Angle arcs
        arc_radius = r * 0.3
        theta_angles = np.linspace(0, theta, 20)
        camera_xy_norm = np.linalg.norm(camera_pos[:2])
        phi_unit = camera_pos[:2] / camera_xy_norm if camera_xy_norm > 0 else np.array([1, 0])
        ax.plot(arc_radius * np.sin(theta_angles) * phi_unit[0],
                arc_radius * np.sin(theta_angles) * phi_unit[1],
                arc_radius * np.cos(theta_angles), 'orange', linewidth=2.5, alpha=0.8)
        mid_theta = theta / 2
        label_r = arc_radius * 1.3
        ax.text(label_r * np.sin(mid_theta) * phi_unit[0], label_r * np.sin(mid_theta) * phi_unit[1],
                label_r * np.cos(mid_theta), f'θ={np.degrees(theta):.1f}°', fontsize=11, color='orange',
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='orange', alpha=0.8))
        
        if camera_xy_norm > 1e-6:
            phi_angles = np.linspace(0, phi, 20)
            phi_arc_radius = arc_radius * 0.8
            ax.plot(phi_arc_radius * np.cos(phi_angles), phi_arc_radius * np.sin(phi_angles),
                    np.zeros(20), 'purple', linewidth=2.5, alpha=0.8)
            mid_phi = phi / 2
            ax.text(phi_arc_radius * 1.4 * np.cos(mid_phi), phi_arc_radius * 1.4 * np.sin(mid_phi), 0,
                    f'φ={np.degrees(phi):.1f}°', fontsize=11, color='purple', fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='purple', alpha=0.8))
        
        if show_gaze_samples:
            grid_size = int(np.sqrt(n_gaze_samples))
            for x in np.linspace(-w/2, w/2, grid_size):
                for y in np.linspace(-h/2, h/2, grid_size):
                    ax.plot([0, x], [0, y], [0, d], 'c-', linewidth=0.5, alpha=0.3)
                    ax.scatter(x, y, d, color='cyan', s=20, alpha=0.5)
        
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f'Experimental Setup: θ={np.degrees(theta):.1f}°, φ={np.degrees(phi):.1f}°, r={r:.0f}mm, d={d:.0f}mm',
                    fontsize=12, fontweight='bold')
        max_range = max(r, d, w/2, h/2) * 1.2
        ax.set_xlim([-max_range, max_range]); ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range/4, d + max_range/4])
        ax.legend(loc='upper left', fontsize=9)
        
        arrow_length = max_range / 4
        ax.quiver(0, 0, 0, arrow_length, 0, 0, color='r', alpha=0.3, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, arrow_length, 0, color='g', alpha=0.3, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, arrow_length, color='b', alpha=0.3, arrow_length_ratio=0.1)
        
        if len(viewing_angle) == 2:
            ax.view_init(elev=viewing_angle[0], azim=viewing_angle[1])
        elif len(viewing_angle) == 3:
            ax.view_init(elev=viewing_angle[0], azim=viewing_angle[1], roll=viewing_angle[2])
        plt.tight_layout()
        return fig, ax
    
    def _plot_2d(self, theta, phi, r, d, w, h, show_gaze_samples, n_gaze_samples):
        """2D orthogonal projections of experimental setup."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Arc
        
        E = np.array([0, 0, 0])
        C = np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])
        screen_corners = np.array([[-w/2, -h/2, d], [w/2, -h/2, d], [w/2, h/2, d], [-w/2, h/2, d], [-w/2, -h/2, d]])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        eye_radius = max(r, d, w/2, h/2) * 0.03
        cam_size = eye_radius * 1.5
        
        # Front View (x-y)
        ax = axes[0]
        ax.set_aspect('equal'); ax.axhline(0, color='k', linestyle=':', alpha=0.3); ax.axvline(0, color='k', linestyle=':', alpha=0.3)
        ax.plot(screen_corners[:, 0], screen_corners[:, 1], 'k-', linewidth=2)
        ax.fill(screen_corners[:-1, 0], screen_corners[:-1, 1], color='lightgray', alpha=0.3)
        ax.add_patch(plt.Circle((E[0], E[1]), eye_radius, color='steelblue', ec='black', linewidth=2, zorder=10))
        ax.plot(E[0], E[1], 'k.', markersize=6, zorder=11)
        ax.text(E[0], E[1] - eye_radius*1.5, 'E', ha='center', va='top', fontsize=10, fontweight='bold')
        ax.add_patch(mpatches.FancyBboxPatch((C[0]-cam_size*0.6, C[1]-cam_size*0.4), cam_size*1.2, cam_size*0.8,
            boxstyle="round,pad=0.05", ec='black', fc='darkred', linewidth=2, zorder=10, alpha=0.7))
        ax.plot(C[0], C[1], 'wo', markersize=8, zorder=11)
        ax.text(C[0], C[1] - cam_size*0.7, 'C', ha='center', va='top', fontsize=10, fontweight='bold', color='white')
        ax.plot([E[0], C[0]], [E[1], C[1]], 'r--', alpha=0.5, linewidth=1.5)
        camera_xy_dist = np.sqrt(C[0]**2 + C[1]**2)
        if camera_xy_dist > 1e-6:
            arc_radius = min(camera_xy_dist, max(w, h)) * 0.3
            phi_deg = np.degrees(phi)
            theta1, theta2 = (0, phi_deg) if phi_deg > 0 else (phi_deg, 0)
            ax.add_patch(Arc((E[0], E[1]), 2*arc_radius, 2*arc_radius, angle=0, theta1=theta1, theta2=theta2, color='purple', linewidth=2.5, zorder=5))
            mid_phi = phi / 2; label_r = arc_radius * 1.3
            ax.text(E[0] + label_r * np.cos(mid_phi), E[1] + label_r * np.sin(mid_phi), f'φ={phi_deg:.1f}°',
                   fontsize=10, color='purple', fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='purple', alpha=0.9))
        if show_gaze_samples:
            n = int(np.sqrt(n_gaze_samples))
            for x_pos in np.linspace(-w/2*0.8, w/2*0.8, n):
                for y_pos in np.linspace(-h/2*0.8, h/2*0.8, n):
                    ax.plot([E[0], x_pos], [E[1], y_pos], 'g-', alpha=0.2, linewidth=0.5)
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_title('Front View (x-y plane)', fontweight='bold'); ax.grid(True, alpha=0.2)
        
        # Top View (z-x)
        ax = axes[1]
        ax.set_aspect('equal'); ax.axhline(0, color='k', linestyle=':', alpha=0.3); ax.axvline(0, color='k', linestyle=':', alpha=0.3)
        ax.plot(screen_corners[:, 2], screen_corners[:, 0], 'k-', linewidth=2)
        ax.add_patch(plt.Circle((E[2], E[0]), eye_radius, color='steelblue', ec='black', linewidth=2, zorder=10))
        ax.plot(E[2], E[0], 'k.', markersize=6, zorder=11)
        ax.text(E[2] - eye_radius*1.5, E[0], 'E', ha='right', va='center', fontsize=10, fontweight='bold')
        ax.add_patch(mpatches.FancyBboxPatch((C[2]-cam_size*0.4, C[0]-cam_size*0.6), cam_size*0.8, cam_size*1.2,
            boxstyle="round,pad=0.05", ec='black', fc='darkred', linewidth=2, zorder=10, alpha=0.7))
        ax.plot(C[2], C[0], 'wo', markersize=8, zorder=11)
        ax.text(C[2], C[0] - cam_size*0.7, 'C', ha='center', va='top', fontsize=10, fontweight='bold', color='white')
        ax.plot([E[2], C[2]], [E[0], C[0]], 'r--', alpha=0.5, linewidth=1.5, label=f'E-C: {r:.0f}mm')
        ax.plot([E[2], d], [E[0], 0], 'g--', alpha=0.5, linewidth=1.5, label=f'E-S: {d:.0f}mm')
        theta_zx = np.arctan2(C[0], C[2])
        if abs(theta_zx) > 1e-6:
            arc_radius = min(r, d) * 0.3; theta_zx_deg = np.degrees(theta_zx)
            theta1, theta2 = (0, theta_zx_deg) if theta_zx_deg > 0 else (theta_zx_deg, 0)
            ax.add_patch(Arc((E[2], E[0]), 2*arc_radius, 2*arc_radius, angle=0, theta1=theta1, theta2=theta2, color='orange', linewidth=2.5, zorder=5))
            mid_angle = theta_zx / 2; label_r = arc_radius * 1.4
            ax.text(E[2] + label_r * np.cos(mid_angle), E[0] + label_r * np.sin(mid_angle), f'θ(x)={theta_zx_deg:.1f}°',
                   fontsize=9, color='orange', fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='orange', alpha=0.9))
        ax.set_xlabel('z (mm)'); ax.set_ylabel('x (mm)'); ax.set_title('Top View (z-x plane)', fontweight='bold'); ax.grid(True, alpha=0.2); ax.legend(loc='upper left', fontsize=8)
        
        # Side View (z-y)
        ax = axes[2]
        ax.set_aspect('equal'); ax.axhline(0, color='k', linestyle=':', alpha=0.3); ax.axvline(0, color='k', linestyle=':', alpha=0.3)
        ax.plot(screen_corners[:, 2], screen_corners[:, 1], 'k-', linewidth=2)
        ax.add_patch(plt.Circle((E[2], E[1]), eye_radius, color='steelblue', ec='black', linewidth=2, zorder=10))
        ax.plot(E[2], E[1], 'k.', markersize=6, zorder=11)
        ax.text(E[2] - eye_radius*1.5, E[1], 'E', ha='right', va='center', fontsize=10, fontweight='bold')
        ax.add_patch(mpatches.FancyBboxPatch((C[2]-cam_size*0.4, C[1]-cam_size*0.6), cam_size*0.8, cam_size*1.2,
            boxstyle="round,pad=0.05", ec='black', fc='darkred', linewidth=2, zorder=10, alpha=0.7))
        ax.plot(C[2], C[1], 'wo', markersize=8, zorder=11)
        ax.text(C[2], C[1] - cam_size*0.7, 'C', ha='center', va='top', fontsize=10, fontweight='bold', color='white')
        ax.plot([E[2], C[2]], [E[1], C[1]], 'r--', alpha=0.5, linewidth=1.5)
        ax.plot([E[2], d], [E[1], 0], 'g--', alpha=0.5, linewidth=1.5)
        theta_zy = np.arctan2(C[1], C[2])
        if abs(theta_zy) > 1e-6:
            arc_radius = min(r, d) * 0.3; theta_zy_deg = np.degrees(theta_zy)
            theta1, theta2 = (0, theta_zy_deg) if theta_zy_deg > 0 else (theta_zy_deg, 0)
            ax.add_patch(Arc((E[2], E[1]), 2*arc_radius, 2*arc_radius, angle=0, theta1=theta1, theta2=theta2, color='orange', linewidth=2.5, zorder=5))
            mid_angle = theta_zy / 2; label_r = arc_radius * 1.4
            ax.text(E[2] + label_r * np.cos(mid_angle), E[1] + label_r * np.sin(mid_angle), f'θ(y)={theta_zy_deg:.1f}°',
                   fontsize=9, color='orange', fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='orange', alpha=0.9))
        ax.set_xlabel('z (mm)'); ax.set_ylabel('y (mm)'); ax.set_title('Side View (z-y plane)', fontweight='bold'); ax.grid(True, alpha=0.2)
        
        fig.suptitle(f'Experimental Setup: θ={np.degrees(theta):.1f}°, φ={np.degrees(phi):.1f}°', fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig, axes
    
    # =========================================================================
    # String representation
    # =========================================================================
    
    def __repr__(self) -> str:
        """Concise representation."""
        parts = ["ExperimentalSetup("]
        
        if self._screen_resolution is not None:
            parts.append(f"  screen={self._screen_resolution[0]}x{self._screen_resolution[1]}px")
        
        if self._physical_screen_size is not None:
            parts.append(f"  physical={self._physical_screen_size[0]:.0f}x{self._physical_screen_size[1]:.0f}mm")
        
        if self._d is not None:
            parts.append(f"  d={self._d:.0f}mm")
        
        if self._screen_offset != (0.0, 0.0):
            parts.append(f"  screen_offset=({self._screen_offset[0]:.1f}, {self._screen_offset[1]:.1f})mm")
        
        if self._alpha_tilt != 0.0 or self._beta_tilt != 0.0:
            parts.append(f"  tilt=({np.degrees(self._alpha_tilt):.1f}°, {np.degrees(self._beta_tilt):.1f}°)")
        
        if self.has_camera_position():
            parts.append(f"  camera: r={self.r:.0f}mm, θ={np.degrees(self.theta):.1f}°, φ={np.degrees(self.phi):.1f}°")
        
        if self._ipd is not None:
            parts.append(f"  ipd={self._ipd:.0f}mm")
        
        if len(parts) == 1:
            parts.append("  <empty>")
        
        parts.append(")")
        return "\n".join(parts)
    
    def __str__(self) -> str:
        """Human-readable string."""
        return self.__repr__()

