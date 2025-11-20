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
    
    def plot(self, show_surface=True, interpolation='rbf'):
        """
        Plot calibration accuracy with target points, measured points, and optional error surface.
        
        Shows target points as black crosses, measured points as colored plus markers
        overlaid on an optional interpolated error surface.
        
        Parameters
        ----------
        show_surface : bool, default True
            If True, show interpolated error surface as background.
            If False, show only the calibration points.
        interpolation : str, default 'rbf'
            Interpolation method for surface (only used if show_surface=True). Options:
            - 'rbf': Radial basis function (smooth, good for scattered data)
            - 'linear', 'cubic', 'nearest': griddata methods
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axis object used for plotting
        
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pypillometry.eyedata.spatial_calibration import SpatialCalibration
        >>> 
        >>> points = np.array([[640, 512], [640, 87], [640, 936],
        ...                    [96, 512], [1183, 512], [161, 138],
        ...                    [1118, 138], [161, 885], [1118, 885]])
        >>> errors = np.array([[0.3, -13, -2], [0.3, 13, -0.2], [0.42, 16, 9],
        ...                    [0.79, -19, 30], [0.62, 13, 24], [0.69, 22, 22],
        ...                    [0.5, 22, 0.8], [0.05, -2, 0.6], [0.24, -9, -6]])
        >>> measured = points + errors[:, 1:]
        >>> 
        >>> cal = SpatialCalibration('left', 'HV9', points, measured, errors, (1920, 1080))
        >>> 
        >>> # Plot with surface
        >>> fig, ax = cal.plot()  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP
        >>> 
        >>> # Plot without surface
        >>> fig, ax = cal.plot(show_surface=False)  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot error surface if requested
        if show_surface:
            if self.screen_resolution is None:
                raise ValueError("screen_resolution must be set to plot surface")
            
            # Create grid for interpolation
            grid_x, grid_y = np.meshgrid(
                np.linspace(0, self.screen_resolution[0], 100),
                np.linspace(0, self.screen_resolution[1], 100)
            )
            
            # Interpolate error values
            if interpolation == 'rbf':
                from scipy.interpolate import Rbf
                rbf = Rbf(self.points[:, 0], self.points[:, 1], 
                         self.errors[:, 0], function='multiquadric', smooth=0.1)
                grid_z = rbf(grid_x, grid_y)
            else:
                from scipy.interpolate import griddata
                grid_z = griddata(
                    self.points, self.errors[:, 0],
                    (grid_x, grid_y), method=interpolation
                )
            
            # Plot surface
            im = ax.imshow(grid_z, extent=[0, self.screen_resolution[0],
                                             self.screen_resolution[1], 0],
                           aspect='auto', cmap='YlOrRd', alpha=0.6, zorder=1)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Error (degrees)', rotation=270, labelpad=15)
        
        # Plot target points as black crosses
        ax.scatter(self.points[:, 0], self.points[:, 1], 
                   marker='x', s=150, c='black', linewidths=2,
                   label='Target', zorder=4)
        
        # Plot measured points as dark blue plus markers
        ax.scatter(self.measured[:, 0], self.measured[:, 1],
                   c='darkblue', s=250, marker='+', linewidths=3,
                   label='Measured', zorder=5)
        
        # Labels and limits
        ax.set_xlabel('X position (pixels)')
        ax.set_ylabel('Y position (pixels)')
        ax.set_title(f'{self.eye.capitalize()} Eye Calibration ({self.model})')
        
        if self.screen_resolution is not None:
            ax.set_xlim(0, self.screen_resolution[0])
            ax.set_ylim(0, self.screen_resolution[1])
            ax.invert_yaxis()  # Match screen coordinates
        
        ax.legend(loc='best')
        
        return fig, ax
    
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

