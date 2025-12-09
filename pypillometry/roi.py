"""
roi.py
=====

Implement region of interest (ROI) functionality for gaze-data (x/y).

Coordinates are stored internally in pixels, with the origin at the upper-left
corner of the screen (standard screen coordinate system, y increases downward).
Alternative coordinate systems can be used via factory methods (`from_mm`, 
`from_degrees`) when an `ExperimentalSetup` is provided.

Coordinate Systems
------------------
- **Pixels**: Origin at upper-left, x increases right, y increases down.
- **Millimeters (mm)**: Origin at screen center, x increases right, y increases up.
- **Degrees**: Visual angles from screen center, x increases right, y increases up.
  Requires eye-to-screen distance from ExperimentalSetup.
"""

from typing import Dict, Union, Tuple, Optional, TYPE_CHECKING
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pypillometry.units import parse_angle

if TYPE_CHECKING:
    from pypillometry.eyedata import ExperimentalSetup

class ROI:
    """Base class for Region of Interest (ROI).
    
    This class provides common functionality for different types of ROIs.
    Subclasses should implement the `__contains__` method to define their specific
    containment logic.
    
    Coordinates are specified in pixels with origin at the upper-left corner
    of the screen (y increases downward). Use factory methods `from_mm()` or
    `from_degrees()` on subclasses for alternative coordinate systems.
    
    Parameters
    ----------
    name : str, optional
        Name of the ROI
    setup : ExperimentalSetup, optional
        Experimental setup for coordinate conversions. If provided, enables
        coordinate system conversion methods.
    """
    
    def __init__(self, name: str = None, setup: Optional["ExperimentalSetup"] = None):
        """Initialize ROI with optional name and setup.
        
        Parameters
        ----------
        name : str, optional
            Name of the ROI, by default None
        setup : ExperimentalSetup, optional
            Experimental setup for coordinate conversions
        """
        self.name = name
        self.setup = setup
    
    def __contains__(self, coords: Union[Tuple[float, float], np.ndarray]) -> bool:
        """Test if a single coordinate is within the ROI (for use with `in`).
        Raises ValueError if an array is passed.
        """
        coords = np.asarray(coords)
        if coords.ndim != 1:
            raise ValueError("Use the .contains() method for array input.")
        return self.contains(coords)

    def contains(self, coords: Union[Tuple[float, float], np.ndarray]) -> Union[bool, np.ndarray]:
        """Test if coordinates are within the ROI.
        Supports both single coordinate and array of coordinates.
        """
        raise NotImplementedError("Subclasses must implement contains()")
    
    def plot(self, ax=None, **kwargs):
        """Plot the ROI on the given axes.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, the current axes will be used.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.
            
        Returns
        -------
        matplotlib.artist.Artist
            The artist object created for the ROI.
        """
        raise NotImplementedError("Subclasses must implement plot()")
    
    def __str__(self) -> str:
        """String representation of the ROI."""
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        return str(self)


class CircularROI(ROI):
    """Circular Region of Interest defined by center point and radius.
    
    Coordinates are in pixels (origin at upper-left, y increases downward) by default.
    Use `from_mm()` or `from_degrees()` factory methods for alternative coordinate systems.
    """
    
    def __init__(
        self, 
        center: Tuple[float, float], 
        radius: float, 
        name: str = None,
        setup: Optional["ExperimentalSetup"] = None
    ):
        """Initialize circular ROI.
        
        Parameters
        ----------
        center : Tuple[float, float]
            (x, y) coordinates of the center point in pixels.
            Origin at upper-left corner, y increases downward.
        radius : float
            Radius of the circle in pixels
        name : str, optional
            Name of the ROI, by default None
        setup : ExperimentalSetup, optional
            Experimental setup for coordinate conversions
            
        Raises
        ------
        ValueError
            If radius is negative
        """
        super().__init__(name, setup)
        self.center = np.array(center)
        if radius < 0:
            raise ValueError("Radius cannot be negative")
        self.radius = float(radius)
    
    @classmethod
    def from_mm(
        cls,
        center_mm: Tuple[float, float],
        radius_mm: float,
        setup: "ExperimentalSetup",
        name: str = None
    ) -> "CircularROI":
        """Create circular ROI from mm coordinates (origin at screen center).
        
        Parameters
        ----------
        center_mm : Tuple[float, float]
            (x, y) center in mm relative to screen center.
            Positive x = right, positive y = up.
        radius_mm : float
            Radius in mm
        setup : ExperimentalSetup
            Experimental setup with screen geometry (screen_resolution and 
            physical_screen_size must be set)
        name : str, optional
            Name of the ROI
        
        Returns
        -------
        CircularROI
            ROI with coordinates converted to pixels
            
        Examples
        --------
        >>> setup = ExperimentalSetup(
        ...     screen_resolution=(1920, 1080),
        ...     physical_screen_size=("52 cm", "29 cm")
        ... )
        >>> roi = CircularROI.from_mm(center_mm=(50, 25), radius_mm=30, setup=setup)
        """
        # Validate setup
        if not setup.has_screen_info():
            raise ValueError(
                "ExperimentalSetup must have screen_resolution and physical_screen_size set."
            )
        
        # Convert center from mm (centered, +y up) to pixels (+y down)
        # mm_to_pixels expects centered coordinates and returns pixel coordinates
        cx_px, cy_px = setup.mm_to_pixels(center_mm[0], -center_mm[1], centered=True)
        
        # Convert radius (use average of x/y scale factors for non-square pixels)
        avg_mm_per_pixel = (setup.mm_per_pixel_x + setup.mm_per_pixel_y) / 2
        radius_px = radius_mm / avg_mm_per_pixel
        
        return cls((cx_px, cy_px), radius_px, name=name, setup=setup)
    
    @classmethod
    def from_degrees(
        cls,
        center: Tuple[Union[float, str], Union[float, str]],
        radius: Union[float, str],
        setup: "ExperimentalSetup",
        name: str = None
    ) -> "CircularROI":
        """Create circular ROI from visual angle coordinates.
        
        Parameters
        ----------
        center : Tuple[float or str, float or str]
            (x, y) center in visual angle from screen center.
            Positive x = right, positive y = up.
            Each coordinate can be:
            - float: assumed to be radians
            - str: with units, e.g., "5 deg", "0.087 rad"
            - pint Quantity
        radius : float or str
            Radius in visual angle. Can be:
            - float: assumed to be radians
            - str: with units, e.g., "2 deg", "0.035 rad"
            - pint Quantity
        setup : ExperimentalSetup
            Experimental setup with screen geometry and eye distance
            (screen_resolution, physical_screen_size, and eye_to_screen_center 
            must be set)
        name : str, optional
            Name of the ROI
        
        Returns
        -------
        CircularROI
            ROI with coordinates converted to pixels
            
        Examples
        --------
        >>> setup = ExperimentalSetup(
        ...     screen_resolution=(1920, 1080),
        ...     physical_screen_size=("52 cm", "29 cm"),
        ...     eye_to_screen_center="60 cm"
        ... )
        >>> roi = CircularROI.from_degrees(center=("5 deg", "2.5 deg"), radius="2 deg", setup=setup)
        >>> roi = CircularROI.from_degrees(center=(0.087, 0.044), radius=0.035, setup=setup)  # radians
        """
        # Validate setup
        if not setup.has_screen_info():
            raise ValueError(
                "ExperimentalSetup must have screen_resolution and physical_screen_size set."
            )
        if not setup.has_eye_distance():
            raise ValueError(
                "ExperimentalSetup must have eye_to_screen_center set for degree conversion."
            )
        
        # Parse angles (converts to radians)
        cx_rad = parse_angle(center[0])
        cy_rad = parse_angle(center[1])
        radius_rad = parse_angle(radius)
        
        # Convert center from radians to mm
        cx_mm = setup.d * np.tan(cx_rad)
        cy_mm = setup.d * np.tan(cy_rad)
        
        # Convert radius from radians to mm (at screen center approximation)
        radius_mm = setup.d * np.tan(radius_rad)
        
        return cls.from_mm((cx_mm, cy_mm), radius_mm, setup=setup, name=name)
    
    def __contains__(self, coords: Union[Tuple[float, float], np.ndarray]) -> bool:
        coords = np.asarray(coords)
        if coords.ndim != 1:
            raise ValueError("Use the .contains() method for array input.")
        return self.contains(coords)

    def contains(self, coords: Union[Tuple[float, float], np.ndarray]) -> Union[bool, np.ndarray]:
        coords = np.asarray(coords)
        if coords.ndim == 1:
            return np.sqrt(np.sum((coords - self.center)**2)) <= self.radius
        else:
            return np.sqrt(np.sum((coords - self.center)**2, axis=1)) <= self.radius
    
    def plot(self, ax=None, **kwargs):
        """Plot the circular ROI.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, the current axes will be used.
        **kwargs : dict
            Additional keyword arguments passed to patches.Circle.
            Common options include:
            - facecolor: color of the circle fill
            - edgecolor: color of the circle edge
            - alpha: transparency
            - fill: whether to fill the circle
            - label: label for the legend
            
        Returns
        -------
        matplotlib.patches.Circle
            The circle patch object.
        """
        if ax is None:
            ax = plt.gca()
            
        # Set default style if not provided
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = 'none'
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'red'
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5
        if 'label' not in kwargs and self.name:
            kwargs['label'] = self.name
            
        circle = patches.Circle(self.center, self.radius, **kwargs)
        ax.add_patch(circle)
        return circle
    
    def __str__(self) -> str:
        """String representation of the circular ROI."""
        return f"{super().__str__()}(center={self.center}, radius={self.radius})"


class RectangularROI(ROI):
    """Rectangular Region of Interest defined by two corners.
    
    Coordinates are in pixels (origin at upper-left, y increases downward) by default.
    Use `from_mm()` or `from_degrees()` factory methods for alternative coordinate systems.
    """
    
    def __init__(
        self, 
        corner1: Tuple[float, float], 
        corner2: Tuple[float, float], 
        name: str = None,
        setup: Optional["ExperimentalSetup"] = None
    ):
        """Initialize rectangular ROI.
        
        Parameters
        ----------
        corner1 : Tuple[float, float]
            (x, y) coordinates of first corner in pixels.
            Origin at upper-left corner, y increases downward.
        corner2 : Tuple[float, float]
            (x, y) coordinates of second corner in pixels.
            Origin at upper-left corner, y increases downward.
        name : str, optional
            Name of the ROI, by default None
        setup : ExperimentalSetup, optional
            Experimental setup for coordinate conversions
        """
        super().__init__(name, setup)
        self.corner1 = np.array(corner1)
        self.corner2 = np.array(corner2)
        # Calculate min/max coordinates for easier containment testing
        self.min_coords = np.minimum(self.corner1, self.corner2)
        self.max_coords = np.maximum(self.corner1, self.corner2)
    
    @classmethod
    def from_mm(
        cls,
        corner1_mm: Tuple[float, float],
        corner2_mm: Tuple[float, float],
        setup: "ExperimentalSetup",
        name: str = None
    ) -> "RectangularROI":
        """Create rectangular ROI from mm coordinates (origin at screen center).
        
        Parameters
        ----------
        corner1_mm : Tuple[float, float]
            (x, y) first corner in mm relative to screen center.
            Positive x = right, positive y = up.
        corner2_mm : Tuple[float, float]
            (x, y) second corner in mm relative to screen center.
            Positive x = right, positive y = up.
        setup : ExperimentalSetup
            Experimental setup with screen geometry (screen_resolution and 
            physical_screen_size must be set)
        name : str, optional
            Name of the ROI
        
        Returns
        -------
        RectangularROI
            ROI with coordinates converted to pixels
            
        Examples
        --------
        >>> setup = ExperimentalSetup(
        ...     screen_resolution=(1920, 1080),
        ...     physical_screen_size=("52 cm", "29 cm")
        ... )
        >>> # Rectangle from (-50, -25) mm to (50, 25) mm (centered, 100x50 mm)
        >>> roi = RectangularROI.from_mm(
        ...     corner1_mm=(-50, -25), corner2_mm=(50, 25), setup=setup
        ... )
        """
        # Validate setup
        if not setup.has_screen_info():
            raise ValueError(
                "ExperimentalSetup must have screen_resolution and physical_screen_size set."
            )
        
        # Convert corners from mm (centered, +y up) to pixels (+y down)
        c1x_px, c1y_px = setup.mm_to_pixels(corner1_mm[0], -corner1_mm[1], centered=True)
        c2x_px, c2y_px = setup.mm_to_pixels(corner2_mm[0], -corner2_mm[1], centered=True)
        
        return cls((c1x_px, c1y_px), (c2x_px, c2y_px), name=name, setup=setup)
    
    @classmethod
    def from_degrees(
        cls,
        corner1: Tuple[Union[float, str], Union[float, str]],
        corner2: Tuple[Union[float, str], Union[float, str]],
        setup: "ExperimentalSetup",
        name: str = None
    ) -> "RectangularROI":
        """Create rectangular ROI from visual angle coordinates.
        
        Parameters
        ----------
        corner1 : Tuple[float or str, float or str]
            (x, y) first corner in visual angle from screen center.
            Positive x = right, positive y = up.
            Each coordinate can be:
            - float: assumed to be radians
            - str: with units, e.g., "-5 deg", "-0.087 rad"
            - pint Quantity
        corner2 : Tuple[float or str, float or str]
            (x, y) second corner in visual angle from screen center.
            Positive x = right, positive y = up.
            Each coordinate can be:
            - float: assumed to be radians
            - str: with units, e.g., "5 deg", "0.087 rad"
            - pint Quantity
        setup : ExperimentalSetup
            Experimental setup with screen geometry and eye distance
            (screen_resolution, physical_screen_size, and eye_to_screen_center 
            must be set)
        name : str, optional
            Name of the ROI
        
        Returns
        -------
        RectangularROI
            ROI with coordinates converted to pixels
            
        Examples
        --------
        >>> setup = ExperimentalSetup(
        ...     screen_resolution=(1920, 1080),
        ...     physical_screen_size=("52 cm", "29 cm"),
        ...     eye_to_screen_center="60 cm"
        ... )
        >>> # Rectangle from (-5, -2.5) deg to (5, 2.5) deg
        >>> roi = RectangularROI.from_degrees(
        ...     corner1=("-5 deg", "-2.5 deg"), corner2=("5 deg", "2.5 deg"), setup=setup
        ... )
        """
        # Validate setup
        if not setup.has_screen_info():
            raise ValueError(
                "ExperimentalSetup must have screen_resolution and physical_screen_size set."
            )
        if not setup.has_eye_distance():
            raise ValueError(
                "ExperimentalSetup must have eye_to_screen_center set for degree conversion."
            )
        
        # Parse angles (converts to radians)
        c1x_rad = parse_angle(corner1[0])
        c1y_rad = parse_angle(corner1[1])
        c2x_rad = parse_angle(corner2[0])
        c2y_rad = parse_angle(corner2[1])
        
        # Convert corners from radians to mm
        c1_mm = (setup.d * np.tan(c1x_rad), setup.d * np.tan(c1y_rad))
        c2_mm = (setup.d * np.tan(c2x_rad), setup.d * np.tan(c2y_rad))
        
        return cls.from_mm(c1_mm, c2_mm, setup=setup, name=name)
    
    def __contains__(self, coords: Union[Tuple[float, float], np.ndarray]) -> bool:
        coords = np.asarray(coords)
        if coords.ndim != 1:
            raise ValueError("Use the .contains() method for array input.")
        return self.contains(coords)

    def contains(self, coords: Union[Tuple[float, float], np.ndarray]) -> Union[bool, np.ndarray]:
        coords = np.asarray(coords)
        if coords.ndim == 1:
            return np.all(coords >= self.min_coords) and np.all(coords <= self.max_coords)
        else:
            return np.all(coords >= self.min_coords, axis=1) & np.all(coords <= self.max_coords, axis=1)
    
    def plot(self, ax=None, **kwargs):
        """Plot the rectangular ROI.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, the current axes will be used.
        **kwargs : dict
            Additional keyword arguments passed to patches.Rectangle.
            Common options include:
            - facecolor: color of the rectangle fill
            - edgecolor: color of the rectangle edge
            - alpha: transparency
            - fill: whether to fill the rectangle
            - label: label for the legend
            
        Returns
        -------
        matplotlib.patches.Rectangle
            The rectangle patch object.
        """
        if ax is None:
            ax = plt.gca()
            
        # Set default style if not provided
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = 'none'
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'red'
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5
        if 'label' not in kwargs and self.name:
            kwargs['label'] = self.name
            
        width = self.max_coords[0] - self.min_coords[0]
        height = self.max_coords[1] - self.min_coords[1]
        rect = patches.Rectangle(self.min_coords, width, height, **kwargs)
        ax.add_patch(rect)
        return rect
    
    def __str__(self) -> str:
        """String representation of the rectangular ROI."""
        return f"{super().__str__()}(corner1={self.corner1}, corner2={self.corner2})"
