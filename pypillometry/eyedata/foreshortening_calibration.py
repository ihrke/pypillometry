"""
Foreshortening correction for pupil size measurements.

This module implements the foreshortening correction algorithm for pupillometry,
which accounts for the viewing angle between the eye-to-camera vector and the
eye-to-gaze vector. The measured pupil size is foreshortened by a factor of
cos(alpha), where alpha is this viewing angle.
"""

import numpy as np
from scipy.interpolate import BSpline
from typing import Optional, Union, Tuple, Dict
from ..intervals import Intervals


def _compute_cos_alpha_vectorized(
    x: Union[float, np.ndarray], 
    y: Union[float, np.ndarray], 
    theta: float, 
    phi: float, 
    r: float, 
    d: float,
    eye_offset: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Compute foreshortening factor cos(alpha) for given gaze position(s).
    
    The foreshortening factor is the cosine of the angle alpha between the
    eye-to-camera vector (EC) and the eye-to-gaze vector (ET).
    
    Parameters
    ----------
    x : float or array
        Gaze x-coordinate(s) on screen in mm (relative to screen center)
    y : float or array
        Gaze y-coordinate(s) on screen in mm (relative to screen center)
    theta : float
        Camera polar angle from +z axis in radians [0, π/2]
        (0=on viewing axis, π/2=perpendicular/side; camera must be forward of eye)
    phi : float
        Camera azimuthal angle in x-y plane in radians [-π, π]
        (0=right/+x, π/2=up/+y, -π/2=down/-y, ±π=left/-x)
    r : float
        Eye-to-camera distance in mm
    d : float
        Eye-to-screen distance in mm
    eye_offset : float, default 0.0
        Eye x-offset for binocular setups (left eye: -IPD/2, right eye: +IPD/2)
        For monocular, use 0.0
    
    Returns
    -------
    cos_alpha : float or array
        Foreshortening factor(s) in range [0, 1]
    
    Notes
    -----
    Camera position in Cartesian coordinates:
        C = (r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta))
    
    Eye position:
        E = (eye_offset, 0, 0)
    
    Gaze position on screen:
        T = (x, y, d)
    
    cos(alpha) = (EC · ET) / (|EC| * |ET|)
    
    Examples
    --------
    >>> # Camera below screen center, 600mm from eye, screen at 700mm
    >>> theta = np.radians(85)  # Slightly offset from perpendicular
    >>> phi = np.radians(-90)  # Below screen (negative y direction)
    >>> x, y = 0.0, 0.0  # Looking at screen center
    >>> cos_alpha = _compute_cos_alpha_vectorized(x, y, theta, phi, 600, 700)
    >>> print(f"cos(alpha) = {cos_alpha:.3f}")  # doctest: +SKIP
    """
    # Camera position in Cartesian coordinates
    cx = r * np.sin(theta) * np.cos(phi)
    cy = r * np.sin(theta) * np.sin(phi)
    cz = r * np.cos(theta)
    
    # Eye-to-camera vector components
    ec_x = cx - eye_offset
    ec_y = cy
    ec_z = cz
    
    # Eye-to-gaze vector components
    et_x = x - eye_offset
    et_y = y
    et_z = d
    
    # Dot product
    numerator = ec_x * et_x + ec_y * et_y + ec_z * et_z
    
    # Magnitude of EC (should equal r for monocular, slightly different for binocular)
    mag_ec = np.sqrt(ec_x**2 + ec_y**2 + ec_z**2)
    
    # Magnitude of ET
    mag_et = np.sqrt(et_x**2 + et_y**2 + et_z**2)
    
    # Cosine of angle
    cos_alpha = numerator / (mag_ec * mag_et)
    
    return cos_alpha


def _determine_knots(
    t_min: float, 
    t_max: float, 
    knots_per_second: float = 1.0
) -> np.ndarray:
    """
    Automatically determine knot positions for B-spline basis.
    
    Parameters
    ----------
    t_min : float
        Minimum time value (ms)
    t_max : float
        Maximum time value (ms)
    knots_per_second : float, default 1.0
        Desired number of knots per second of data
    
    Returns
    -------
    knots : np.ndarray
        Knot positions including boundary knots for cubic B-splines
    
    Notes
    -----
    For cubic B-splines (degree 3), we need to repeat the boundary knots
    4 times (degree + 1) to ensure the spline interpolates at the boundaries.
    
    Examples
    --------
    >>> knots = _determine_knots(0, 120000, knots_per_second=1.0)  # 2 minutes
    >>> print(f"Number of interior knots: {len(knots) - 8}")  # doctest: +SKIP
    """
    duration_ms = t_max - t_min
    duration_sec = duration_ms / 1000.0
    
    # Number of interior knots
    n_interior_knots = max(int(np.ceil(duration_sec * knots_per_second)), 2)
    
    # Interior knot positions (uniformly spaced)
    interior_knots = np.linspace(t_min, t_max, n_interior_knots)
    
    # For cubic B-splines (degree 3), repeat boundary knots 4 times
    knots = np.concatenate([
        [t_min] * 4,  # Left boundary knots
        interior_knots,  # Interior knots
        [t_max] * 4   # Right boundary knots
    ])
    
    return knots


def _create_bspline_basis(
    t: np.ndarray, 
    knots: np.ndarray, 
    degree: int = 3
) -> np.ndarray:
    """
    Create B-spline basis evaluation matrix.
    
    Parameters
    ----------
    t : np.ndarray
        Time points at which to evaluate basis functions (shape: (n,))
    knots : np.ndarray
        Knot vector including repeated boundary knots
    degree : int, default 3
        Degree of B-spline (3 for cubic)
    
    Returns
    -------
    basis_matrix : np.ndarray
        Matrix where basis_matrix[i, k] = B_k(t[i])
        Shape: (n_timepoints, n_basis_functions)
    
    Notes
    -----
    The number of basis functions is: len(knots) - degree - 1
    
    Examples
    --------
    >>> t = np.linspace(0, 100, 50)
    >>> knots = _determine_knots(0, 100, knots_per_second=1.0)
    >>> basis = _create_bspline_basis(t, knots, degree=3)
    >>> print(f"Basis shape: {basis.shape}")  # doctest: +SKIP
    """
    n_basis = len(knots) - degree - 1
    n_points = len(t)
    
    basis_matrix = np.zeros((n_points, n_basis))
    
    # Evaluate each basis function at all time points
    for k in range(n_basis):
        # Create coefficient vector with 1 at position k
        coeffs = np.zeros(n_basis)
        coeffs[k] = 1.0
        
        # Create BSpline object and evaluate
        spline = BSpline(knots, coeffs, degree, extrapolate=False)
        basis_matrix[:, k] = spline(t)
    
    # Replace NaN with 0 (can happen at boundaries with extrapolate=False)
    basis_matrix = np.nan_to_num(basis_matrix, nan=0.0)
    
    return basis_matrix


def _split_params(
    params: np.ndarray, 
    n_basis: int
) -> Tuple[np.ndarray, float, float]:
    """
    Split parameter vector into spline coefficients and camera angles.
    
    Parameters
    ----------
    params : np.ndarray
        Full parameter vector [a_1, ..., a_K, theta, phi]
    n_basis : int
        Number of basis functions (K)
    
    Returns
    -------
    spline_coeffs : np.ndarray
        B-spline coefficients (length K)
    theta : float
        Camera polar angle
    phi : float
        Camera azimuthal angle
    """
    spline_coeffs = params[:n_basis]
    theta = params[n_basis]
    phi = params[n_basis + 1]
    return spline_coeffs, theta, phi


def _objective_function(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    A: np.ndarray,
    basis_matrix: np.ndarray,
    r: float,
    d: float,
    lambda_smooth: float,
    eye_offset: float = 0.0
) -> float:
    """
    Objective function for foreshortening model optimization.
    
    Computes the sum of squared residuals between measured and predicted
    pupil sizes, plus a smoothness regularization term on the spline coefficients.
    
    Parameters
    ----------
    params : np.ndarray
        Parameter vector [a_1, ..., a_K, theta, phi]
    x : np.ndarray
        Gaze x-coordinates in mm (shape: (n,))
    y : np.ndarray
        Gaze y-coordinates in mm (shape: (n,))
    t : np.ndarray
        Time points in ms (shape: (n,))
    A : np.ndarray
        Measured pupil sizes (shape: (n,))
    basis_matrix : np.ndarray
        B-spline basis evaluation matrix (shape: (n, K))
    r : float
        Eye-to-camera distance in mm
    d : float
        Eye-to-screen distance in mm
    lambda_smooth : float
        Smoothness regularization weight
    eye_offset : float, default 0.0
        Eye x-offset for binocular setups
    
    Returns
    -------
    loss : float
        Total loss (data fidelity + smoothness regularization)
    
    Notes
    -----
    Model: A_predicted = (sum_k a_k * B_k(t)) * cos(alpha(x, y; theta, phi))
    
    Loss = sum_i (A_i - A_predicted_i)^2 + lambda_smooth * sum_k (a_{k+1} - a_k)^2
    """
    n_basis = basis_matrix.shape[1]
    spline_coeffs, theta, phi = _split_params(params, n_basis)
    
    # Compute A0(t) from spline
    A0 = basis_matrix @ spline_coeffs
    
    # Compute foreshortening factor
    cos_alpha = _compute_cos_alpha_vectorized(x, y, theta, phi, r, d, eye_offset)
    
    # Predicted measured pupil size
    A_pred = A0 * cos_alpha
    
    # Data fidelity term
    residuals = A - A_pred
    data_loss = np.sum(residuals ** 2)
    
    # Smoothness regularization
    spline_diffs = np.diff(spline_coeffs)
    smooth_loss = lambda_smooth * np.sum(spline_diffs ** 2)
    
    total_loss = data_loss + smooth_loss
    
    return total_loss


def _objective_gradient(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    A: np.ndarray,
    basis_matrix: np.ndarray,
    r: float,
    d: float,
    lambda_smooth: float,
    eye_offset: float = 0.0
) -> np.ndarray:
    """
    Gradient of objective function for faster optimization.
    
    Computes analytical gradient of the loss with respect to all parameters.
    
    Parameters
    ----------
    params : np.ndarray
        Parameter vector [a_1, ..., a_K, theta, phi]
    x, y, t, A : np.ndarray
        Data arrays
    basis_matrix : np.ndarray
        B-spline basis matrix
    r, d : float
        Geometric parameters
    lambda_smooth : float
        Smoothness weight
    eye_offset : float, default 0.0
        Eye x-offset
    
    Returns
    -------
    gradient : np.ndarray
        Gradient vector (same shape as params)
    
    Notes
    -----
    Using chain rule:
    dL/da_k = -2 * sum_i (A_i - A_pred_i) * cos_alpha_i * B_k(t_i) + smoothness_gradient
    dL/dtheta = -2 * sum_i (A_i - A_pred_i) * A0_i * d(cos_alpha)/dtheta
    dL/dphi = -2 * sum_i (A_i - A_pred_i) * A0_i * d(cos_alpha)/dphi
    """
    n_basis = basis_matrix.shape[1]
    spline_coeffs, theta, phi = _split_params(params, n_basis)
    
    # Forward pass
    A0 = basis_matrix @ spline_coeffs
    cos_alpha = _compute_cos_alpha_vectorized(x, y, theta, phi, r, d, eye_offset)
    A_pred = A0 * cos_alpha
    residuals = A - A_pred  # Shape: (n,)
    
    # Gradient w.r.t. spline coefficients
    grad_coeffs = -2 * (basis_matrix.T @ (residuals * cos_alpha))  # Shape: (K,)
    
    # Add smoothness gradient
    # d/da_k of sum_{j} (a_{j+1} - a_j)^2
    smooth_grad = np.zeros(n_basis)
    smooth_grad[:-1] += -2 * lambda_smooth * np.diff(spline_coeffs)
    smooth_grad[1:] += 2 * lambda_smooth * np.diff(spline_coeffs)
    grad_coeffs += smooth_grad
    
    # Gradient w.r.t. theta and phi (numerical approximation for simplicity)
    eps = 1e-6
    
    # Theta gradient
    cos_alpha_theta_plus = _compute_cos_alpha_vectorized(x, y, theta + eps, phi, r, d, eye_offset)
    A_pred_theta_plus = A0 * cos_alpha_theta_plus
    residuals_theta_plus = A - A_pred_theta_plus
    grad_theta = (np.sum(residuals_theta_plus ** 2) - np.sum(residuals ** 2)) / eps
    
    # Phi gradient
    cos_alpha_phi_plus = _compute_cos_alpha_vectorized(x, y, theta, phi + eps, r, d, eye_offset)
    A_pred_phi_plus = A0 * cos_alpha_phi_plus
    residuals_phi_plus = A - A_pred_phi_plus
    grad_phi = (np.sum(residuals_phi_plus ** 2) - np.sum(residuals ** 2)) / eps
    
    # Combine gradients
    gradient = np.concatenate([grad_coeffs, [grad_theta], [grad_phi]])
    
    return gradient


class ForeshorteningCalibration:
    """
    Container for foreshortening correction calibration parameters.
    
    This class stores the fitted camera geometry and temporal pupil model
    needed to correct pupil size measurements for viewing angle effects.
    
    Attributes
    ----------
    eye : str
        Which eye ('left' or 'right')
    theta : float
        Camera polar angle in radians (0 to pi)
    phi : float
        Camera azimuthal angle in radians (0 to 2*pi)
    r : float
        Eye-to-camera distance in mm
    d : float
        Eye-to-screen distance in mm
    spline_coeffs : np.ndarray
        B-spline coefficients for true pupil size A0(t)
    spline_knots : np.ndarray
        Knot vector for B-spline
    spline_degree : int
        Degree of B-spline (typically 3 for cubic)
    fit_intervals : Intervals or None
        Intervals used for fitting, if any
    fit_metrics : dict
        Fit quality metrics (R², RMSE, etc.)
    
    Examples
    --------
    >>> import numpy as np
    >>> from pypillometry.eyedata.foreshortening_calibration import ForeshorteningCalibration
    >>> 
    >>> # Create a calibration object with example parameters
    >>> theta = np.radians(95)
    >>> phi = 0.0
    >>> knots = np.array([0]*4 + [0, 50, 100] + [100]*4)
    >>> coeffs = np.array([3.5, 3.7, 3.6, 3.8, 3.5])
    >>> 
    >>> cal = ForeshorteningCalibration(
    ...     eye='left',
    ...     theta=theta,
    ...     phi=phi,
    ...     r=600.0,
    ...     d=700.0,
    ...     spline_coeffs=coeffs,
    ...     spline_knots=knots,
    ...     spline_degree=3,
    ...     fit_metrics={'r2': 0.95, 'rmse': 0.12}
    ... )
    >>> 
    >>> # Compute foreshortening factor at screen center
    >>> cos_alpha = cal.compute_cos_alpha(0, 0)
    >>> print(f"cos(alpha) at center: {cos_alpha:.3f}")  # doctest: +SKIP
    >>> 
    >>> # Get correction factor
    >>> correction = cal.get_correction_factor(0, 0)
    >>> print(f"Correction factor: {correction:.3f}")  # doctest: +SKIP
    """
    
    def __init__(
        self,
        eye: str,
        theta: float,
        phi: float,
        r: float,
        d: float,
        spline_coeffs: np.ndarray,
        spline_knots: np.ndarray,
        spline_degree: int = 3,
        fit_intervals: Optional[Intervals] = None,
        fit_metrics: Optional[Dict] = None
    ):
        """
        Initialize ForeshorteningCalibration.
        
        Parameters
        ----------
        eye : str
            Which eye ('left' or 'right')
        theta : float
            Camera polar angle in radians
        phi : float
            Camera azimuthal angle in radians
        r : float
            Eye-to-camera distance in mm
        d : float
            Eye-to-screen distance in mm
        spline_coeffs : np.ndarray
            B-spline coefficients
        spline_knots : np.ndarray
            Knot vector
        spline_degree : int, default 3
            Degree of B-spline
        fit_intervals : Intervals or None
            Intervals used for fitting
        fit_metrics : dict or None
            Fit quality metrics
        """
        self.eye = eye
        self.theta = theta
        self.phi = phi
        self.r = r
        self.d = d
        self.spline_coeffs = np.asarray(spline_coeffs)
        self.spline_knots = np.asarray(spline_knots)
        self.spline_degree = spline_degree
        self.fit_intervals = fit_intervals
        self.fit_metrics = fit_metrics if fit_metrics is not None else {}
        
        # Create BSpline object for efficient evaluation
        self._spline = BSpline(
            self.spline_knots, 
            self.spline_coeffs, 
            self.spline_degree,
            extrapolate=True
        )
    
    def compute_cos_alpha(
        self, 
        x: Union[float, np.ndarray], 
        y: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute foreshortening factor cos(alpha) for given gaze position(s).
        
        Parameters
        ----------
        x : float or array
            Gaze x-coordinate(s) in mm
        y : float or array
            Gaze y-coordinate(s) in mm
        
        Returns
        -------
        cos_alpha : float or array
            Foreshortening factor(s)
        
        Examples
        --------
        >>> cos_alpha = cal.compute_cos_alpha(0, 0)  # doctest: +SKIP
        >>> cos_alpha_array = cal.compute_cos_alpha([0, 100], [0, 50])  # doctest: +SKIP
        """
        eye_offset = 0.0  # Monocular for now
        return _compute_cos_alpha_vectorized(
            x, y, self.theta, self.phi, self.r, self.d, eye_offset
        )
    
    def evaluate_spline(
        self, 
        t: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Evaluate true pupil size A0(t) at given time point(s).
        
        Parameters
        ----------
        t : float or array
            Time point(s) in ms
        
        Returns
        -------
        A0 : float or array
            True pupil size(s)
        
        Examples
        --------
        >>> A0_at_t50 = cal.evaluate_spline(50.0)  # doctest: +SKIP
        >>> A0_trace = cal.evaluate_spline(np.linspace(0, 100, 100))  # doctest: +SKIP
        """
        return self._spline(t)
    
    def get_correction_factor(
        self, 
        x: Union[float, np.ndarray], 
        y: Union[float, np.ndarray],
        threshold: float = 0.15
    ) -> Union[float, np.ndarray]:
        """
        Compute correction factor 1/cos(alpha) with quality control thresholding.
        
        When cos(alpha) is very small (extreme viewing angles), the correction
        factor becomes very large and amplifies measurement noise. This method
        returns NaN for samples where cos(alpha) < threshold.
        
        Parameters
        ----------
        x : float or array
            Gaze x-coordinate(s) in mm
        y : float or array
            Gaze y-coordinate(s) in mm
        threshold : float, default 0.15
            Minimum cos(alpha) for reliable correction
            (0.15 corresponds to ~81° viewing angle)
        
        Returns
        -------
        correction_factor : float or array
            Correction factor(s), with NaN where cos(alpha) < threshold
        
        Notes
        -----
        To apply correction: A0_corrected = A_measured * correction_factor
        
        Examples
        --------
        >>> correction = cal.get_correction_factor(0, 0)  # doctest: +SKIP
        >>> corrected_pupil = measured_pupil * correction  # doctest: +SKIP
        """
        cos_alpha = self.compute_cos_alpha(x, y)
        
        # Apply threshold for quality control
        correction_factor = np.where(
            cos_alpha >= threshold,
            1.0 / cos_alpha,
            np.nan
        )
        
        return correction_factor
    
    def __repr__(self) -> str:
        """Concise one-line representation."""
        r2 = self.fit_metrics.get('r2', np.nan)
        rmse = self.fit_metrics.get('rmse', np.nan)
        n_coeffs = len(self.spline_coeffs)
        
        return (f"ForeshorteningCalibration(eye='{self.eye}', "
                f"theta={np.degrees(self.theta):.1f}°, "
                f"phi={np.degrees(self.phi):.1f}°, "
                f"n_coeffs={n_coeffs}, R²={r2:.3f}, RMSE={rmse:.3f})")

