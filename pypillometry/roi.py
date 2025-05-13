"""
roi.py
=====

Implement region of interest (ROI) functionality for gaze-data (x/y).
"""

import os
import requests
from typing import Dict, Union, Tuple
import numpy as np
from loguru import logger

class ROI:
    """Base class for Region of Interest (ROI).
    
    This class provides common functionality for different types of ROIs.
    Subclasses should implement the `__contains__` method to define their specific
    containment logic.
    """
    
    def __init__(self, name: str = None):
        """Initialize ROI with optional name.
        
        Parameters
        ----------
        name : str, optional
            Name of the ROI, by default None
        """
        self.name = name
    
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
    
    def __str__(self) -> str:
        """String representation of the ROI."""
        return f"{self.__class__.__name__}(name={self.name})"


class CircularROI(ROI):
    """Circular Region of Interest defined by center point and radius."""
    
    def __init__(self, center: Tuple[float, float], radius: float, name: str = None):
        """Initialize circular ROI.
        
        Parameters
        ----------
        center : Tuple[float, float]
            (x,y) coordinates of the center point
        radius : float
            Radius of the circle
        name : str, optional
            Name of the ROI, by default None
            
        Raises
        ------
        ValueError
            If radius is negative
        """
        super().__init__(name)
        self.center = np.array(center)
        if radius < 0:
            raise ValueError("Radius cannot be negative")
        self.radius = float(radius)
    
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
    
    def __str__(self) -> str:
        """String representation of the circular ROI."""
        return f"{super().__str__()}(center={self.center}, radius={self.radius})"


class RectangularROI(ROI):
    """Rectangular Region of Interest defined by two corners."""
    
    def __init__(self, corner1: Tuple[float, float], corner2: Tuple[float, float], name: str = None):
        """Initialize rectangular ROI.
        
        Parameters
        ----------
        corner1 : Tuple[float, float]
            (x,y) coordinates of first corner
        corner2 : Tuple[float, float]
            (x,y) coordinates of second corner
        name : str, optional
            Name of the ROI, by default None
        """
        super().__init__(name)
        self.corner1 = np.array(corner1)
        self.corner2 = np.array(corner2)
        # Calculate min/max coordinates for easier containment testing
        self.min_coords = np.minimum(self.corner1, self.corner2)
        self.max_coords = np.maximum(self.corner1, self.corner2)
    
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
    
    def __str__(self) -> str:
        """String representation of the rectangular ROI."""
        return f"{super().__str__()}(corner1={self.corner1}, corner2={self.corner2})"
