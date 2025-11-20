"""
Spatial calibration data handling for eye-tracking.

This module provides the SpatialCalibration class for storing and analyzing
spatial calibration/validation data from eye-trackers.
"""

import numpy as np
from typing import Optional, Dict, Tuple


class SpatialCalibration:
    """
    Store and analyze spatial calibration/validation data from eye-tracking.
    
    Attributes
    ----------
    eye : str
        Eye identifier ('left', 'right', 'binocular')
    model : str
        Calibration model (e.g., 'HV9', 'HV13')
    points : np.ndarray
        Calibration target points (N x 2): [x, y] in pixels
    measured : np.ndarray
        Measured gaze points (N x 2): [x, y] in pixels
    errors : np.ndarray
        Errors (N x 3): [offset, diff_x, diff_y]
    screen_resolution : tuple or None
        Screen resolution (width, height) in pixels
    
    Examples
    --------
    >>> import numpy as np
    >>> from pypillometry.eyedata.spatial_calibration import SpatialCalibration
    >>> 
    >>> # Create synthetic calibration data
    >>> points = np.array([[640, 512], [640, 87], [640, 936]])
    >>> errors = np.array([[0.3, -13.3, -1.6], [0.3, 13.2, -0.2], [0.42, 16.1, 9.2]])
    >>> measured = points + errors[:, 1:]
    >>> 
    >>> cal = SpatialCalibration(
    ...     eye='left',
    ...     model='HV9',
    ...     points=points,
    ...     measured=measured,
    ...     errors=errors,
    ...     screen_resolution=(1920, 1080)
    ... )
    >>> 
    >>> print(cal.eye)
    left
    >>> print(cal.model)
    HV9
    >>> len(cal.points)
    3
    """
    
    def __init__(
        self,
        eye: str,
        model: str,
        points: np.ndarray,
        measured: np.ndarray,
        errors: np.ndarray,
        screen_resolution: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize SpatialCalibration object.
        
        Parameters
        ----------
        eye : str
            Eye identifier
        model : str
            Calibration model name
        points : np.ndarray
            Target points (N x 2)
        measured : np.ndarray
            Measured gaze points (N x 2)
        errors : np.ndarray
            Error values (N x 3): [offset, diff_x, diff_y]
        screen_resolution : tuple, optional
            Screen resolution (width, height) in pixels
        """
        self.eye = eye
        self.model = model
        self.points = points
        self.measured = measured
        self.errors = errors
        self.screen_resolution = screen_resolution
    
    @classmethod
    def from_eyelink(
        cls,
        calibration_data: dict,
        screen_resolution: Optional[Tuple[int, int]] = None
    ) -> 'SpatialCalibration':
        """
        Parse calibration data from eyelinkio format.
        
        Parameters
        ----------
        calibration_data : dict
            Single calibration/validation entry from eyelinkio.
            Expected keys: 'eye', 'model', 'validation'
            validation is structured array with dtype:
            [('point_x', '<f8'), ('point_y', '<f8'), ('offset', '<f8'),
             ('diff_x', '<f8'), ('diff_y', '<f8')]
        screen_resolution : tuple, optional
            Screen resolution (width, height) in pixels
        
        Returns
        -------
        SpatialCalibration
            Parsed calibration object
        
        Examples
        --------
        >>> import numpy as np
        >>> from pypillometry.eyedata.spatial_calibration import SpatialCalibration
        >>> 
        >>> # Create mock EyeLink validation data
        >>> validation_array = np.array([
        ...     (640., 512., 0.3, -13.3, -1.6),
        ...     (640., 87., 0.3, 13.2, -0.2),
        ...     (640., 936., 0.42, 16.1, 9.2)
        ... ], dtype=[('point_x', '<f8'), ('point_y', '<f8'), ('offset', '<f8'),
        ...           ('diff_x', '<f8'), ('diff_y', '<f8')])
        >>> 
        >>> eyelink_data = {
        ...     'eye': 'left',
        ...     'model': 'HV9',
        ...     'validation': validation_array
        ... }
        >>> 
        >>> cal = SpatialCalibration.from_eyelink(eyelink_data, (1920, 1080))
        >>> print(cal.eye)
        left
        >>> print(cal.model)
        HV9
        >>> len(cal.points)
        3
        >>> float(cal.points[0, 0])
        640.0
        """
        val = calibration_data['validation']
        
        # Extract target points
        points = np.column_stack([val['point_x'], val['point_y']])
        
        # Extract errors
        errors = np.column_stack([val['offset'], val['diff_x'], val['diff_y']])
        
        # Calculate measured positions
        measured = points + errors[:, 1:]
        
        return cls(
            eye=calibration_data.get('eye', 'unknown').lower(),
            model=calibration_data.get('model', 'unknown'),
            points=points,
            measured=measured,
            errors=errors,
            screen_resolution=screen_resolution
        )
    
    def get_stats(self) -> Dict[str, float]:
        """
        Calculate calibration error statistics.
        
        Returns
        -------
        dict
            Statistics including: n, mean, sd, max, min (errors in degrees)
        
        Examples
        --------
        >>> import numpy as np
        >>> from pypillometry.eyedata.spatial_calibration import SpatialCalibration
        >>> 
        >>> points = np.array([[640, 512], [640, 87], [640, 936]])
        >>> errors = np.array([[0.3, -13.3, -1.6], [0.5, 13.2, -0.2], [0.42, 16.1, 9.2]])
        >>> measured = points + errors[:, 1:]
        >>> 
        >>> cal = SpatialCalibration('left', 'HV9', points, measured, errors)
        >>> stats = cal.get_stats()
        >>> stats['n']
        3
        >>> f"{stats['mean']:.2f}"
        '0.41'
        >>> f"{stats['max']:.2f}"
        '0.50'
        """
        return {
            'n': len(self.points),
            'mean': float(np.mean(self.errors[:, 0])),
            'sd': float(np.std(self.errors[:, 0])),
            'max': float(np.max(self.errors[:, 0])),
            'min': float(np.min(self.errors[:, 0]))
        }
    
    def __repr__(self) -> str:
        """
        String representation with key statistics.
        
        Examples
        --------
        >>> import numpy as np
        >>> from pypillometry.eyedata.spatial_calibration import SpatialCalibration
        >>> 
        >>> points = np.array([[640, 512], [640, 87]])
        >>> errors = np.array([[0.3, -13.3, -1.6], [0.5, 13.2, -0.2]])
        >>> measured = points + errors[:, 1:]
        >>> 
        >>> cal = SpatialCalibration('left', 'HV9', points, measured, errors)
        >>> print(repr(cal))  # doctest: +ELLIPSIS
        <SpatialCalibration: left eye, HV9, n=2, error=0.40±0.10° (max=0.50°)>
        """
        stats = self.get_stats()
        return (
            f"<SpatialCalibration: {self.eye} eye, {self.model}, "
            f"n={stats['n']}, error={stats['mean']:.2f}±{stats['sd']:.2f}° "
            f"(max={stats['max']:.2f}°)>"
        )

