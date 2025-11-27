from .generic import keephistory
from .gazedata import GazeData
from .eyedatadict import EyeDataDict
from ..plot import EyePlotter
from ..units import parse_distance, parse_angle
import numpy as np
from loguru import logger

from .pupildata import PupilData
import numpy as np
from collections.abc import Iterable
from typing import Optional, Dict, Union
from .spatial_calibration import SpatialCalibration
from .experimental_setup import ExperimentalSetup
from .foreshortening_calibration import (
    ForeshorteningCalibration,
    _determine_knots,
    _create_bspline_basis,
    _objective_function,
    _objective_gradient,
    _split_params
)
from ..intervals import Intervals
from scipy import optimize
from ..signal.baseline import butter_lowpass_filter, downsample



class EyeData(GazeData,PupilData):
    """
    Class for handling eye-tracking data. This class is a subclass of GazeData
    and inherits all its methods and attributes. In addition to the methods
    in GazeData which implement functions for eye-tracking data, `EyeData` 
    provides methods for handling pupillometric data.

    Parameters
    ----------
    time: 
        timing array or `None`, in which case the time-array goes from [0,maxT]
        using `sampling_rate` (in ms)
    left_x, left_y, left_pupil:
        data from left eye (at least one of the eyes must be provided, both x and y)
        pupil is optional
    right_x, right_y, right_pupil:
        data from right eye (at least one of the eyes must be provided, both x and y)
        pupil is optional
    sampling_rate: float
        sampling-rate of the pupillary signal in Hz; if None, 
    experimental_setup: ExperimentalSetup, optional
        Geometric model of the experimental setup including screen geometry,
        eye position, camera position, and screen orientation.
    name: 
        name of the dataset or `None` (in which case a random string is selected)
    event_onsets: 
        time-onsets of any events in the data (in ms, matched in `time` vector)
    event_labels:
        for each event in `event_onsets`, provide a label
    calibration: dict, optional
        Dictionary of SpatialCalibration objects, one per eye (keys: 'left', 'right')
    keep_orig: bool
        keep a copy of the original dataset? If `True`, a copy of the object
        as initiated in the constructor is stored in member `original`
    fill_time_discontinuities: bool
        sometimes, when the eyetracker loses signal, no entry in the EDF is made; 
        when this option is True, such entries will be made and the signal set to 0 there
        (or do it later using `fill_time_discontinuities()`)

    Examples
    --------
    >>> import numpy as np
    >>> from pypillometry.eyedata import EyeData
    >>> d=EyeData(time=np.arange(0, 1000, 1),
    ...         left_x=np.random.randn(1000),
    ...         left_y=np.random.randn(1000))
    >>> print(d)
    """
    def __init__(self, 
                    time: np.ndarray = None,
                    left_x: np.ndarray = None,
                    left_y: np.ndarray = None,
                    left_pupil: np.ndarray = None,
                    right_x: np.ndarray = None,
                    right_y: np.ndarray = None,
                    right_pupil: np.ndarray = None,
                    event_onsets: np.ndarray = None,
                    event_labels: np.ndarray = None,
                    sampling_rate: float = None,
                    experimental_setup: Optional[ExperimentalSetup] = None,
                    name: str = None,
                    calibration: Optional[Dict[str, SpatialCalibration]] = None,
                    fill_time_discontinuities: bool = True,
                    keep_orig: bool = False, 
                    info: dict = None,
                    inplace: bool = False):
        """Constructor for the EyeData class.
        """
        logger.debug("Creating EyeData object")
        if (left_x is None or left_y is None) and (right_x is None or right_y is None):
            raise ValueError("At least one of the eye-traces must be provided (both x and y)")
        self.data=EyeDataDict(left_x=left_x, left_y=left_y, left_pupil=left_pupil,
                                right_x=right_x, right_y=right_y, right_pupil=right_pupil)
        
        
        self._init_common(time, sampling_rate, 
                          event_onsets, event_labels, 
                          name, fill_time_discontinuities, 
                          info=info, inplace=inplace)
        
        # Experimental setup (geometry)
        self.experimental_setup = experimental_setup
        
        # Spatial calibration data
        self.calibration = calibration

        self.original=None
        if keep_orig: 
            self.original=self.copy()

    @property
    def plot(self):
        return EyePlotter(self)


    def summary(self):
        """Return a summary of the dataset as a dictionary.

        Returns
        -------
        : dict
            A dictionary summarizing the dataset.
        """        
        ds = dict(GazeData.summary(self),
                  **PupilData.summary(self))
        return ds

    def get_pupildata(self, eye=None):
        """
        Return the pupil data as a PupilData object.

        Parameters
        ----------
        eye : str, optional
            Which eye to return data for.

        Returns
        -------
        : :class:`PupilData`
            A PupilData object containing the pupil data. 
        """
        if eye is None:
            if len(self.eyes)==1:
                eye=self.eyes[0]
            else:
                raise ValueError("More than one eye in the dataset. Please specify which eye to use.")
        if eye not in [k.split("_")[0] for k in self.data.keys()]:
            raise ValueError("No pupil data for eye: %s" % eye)
        kwargs = {
            "time": self.tx,
            eye+"_pupil": self.data[eye+"_pupil"],
            "sampling_rate": self.fs,
            "event_onsets": self.event_onsets,
            "event_labels": self.event_labels,
            "name": self.name+"_pd",
            "keep_orig": False,
            "inplace": False
        }
        pobj = PupilData(**kwargs)
        return pobj        
    

    @keephistory
    def correct_pupil_foreshortening(self, eyes=[], midpoint=None, store_as="pupil", inplace=None):
        """Correct pupil data using a simple algorithm to correct for foreshortening effects.

        Correct the pupil data for foreshortening effects caused
        by saccades/eye movements. This method is based on a simple algorithm
        described here: 

        :ref:`Correcting pupillary signal using a simple Foreshortening Algorithm </docs/pupil_correction_carole.rst>`

        Relevant publication (not the description of the algorithm used here):    
        https://link.springer.com/article/10.3758/s13428-015-0588-x

        - [ ] TODO: when using interpolated pupil data, x/y data may still be missing. Make sure that the 
          pupil data is not touched where x/y is missing
        

        Parameters 
        ----------
        eyes: list
            Which eyes to correct. If None, correct all available eyes.
        midpoint: tuple
            The center of the screen (x,y) where it is assumed that the pupil is completely circular.
            If None, the midpoint is taken to be the center of the screen as registered
            in the experimental_setup. 
        inplace: bool
            Whether to modify the object in place or return a new object.
            `true`: modify in place
            `false`: return a new object
            `None`: use the object's default setting (initialized in __init__)

        Returns
        -------
        : :class:`EyeData`
            An EyeData object with the corrected pupil
        """
        obj=self._get_inplace(inplace)
        eyes,_=self._get_eye_var(eyes, [])

        if not isinstance(store_as, str):
            logger.warning("store_as must be a string; using 'pupil' instead")
            store_as="pupil"

        # Require experimental_setup for this correction
        if self.experimental_setup is None:
            raise ValueError(
                "Cannot correct pupil foreshortening: experimental_setup not set. "
                "Use set_experimental_setup() to configure it."
            )
        setup = self.experimental_setup

        if midpoint is None:
            midpoint = (setup.screen_width / 2, setup.screen_height / 2)
        
        scaling_factor_x = setup.mm_per_pixel_x
        scaling_factor_y = setup.mm_per_pixel_y

        # calculate distance of x,y from midpoint
        for eye in eyes:
            vx="_".join([eye, "x"])
            vy="_".join([eye, "y"])
            xdist = np.abs(self.data[vx] - midpoint[0]) * scaling_factor_x
            ydist = np.abs(self.data[vy] - midpoint[1]) * scaling_factor_y
            dist = np.sqrt(xdist**2 + ydist**2)
            corr = np.sqrt((dist**2) / (setup.d**2) + 1)  # correction factor
            # Use masked array access to preserve masks from blinks/artifacts
            obj[eye, store_as] = self[eye, "pupil"] * corr

        return obj

    def fit_foreshortening(
        self,
        eye: str,
        r: Optional[Union[float, str]] = None,
        d: Optional[Union[float, str]] = None,
        intervals: Optional[Intervals] = None,
        target_fs: float = 50.0,
        lowpass_freq: float = 4.0,
        lambda_smooth: float = 1.0,
        knots_per_second: float = 1.0,
        initial_theta: Optional[Union[float, str]] = None,
        initial_phi: Optional[Union[float, str]] = None,
        verbose: bool = True
    ) -> ForeshorteningCalibration:
        """
        Fit foreshortening correction model to estimate camera geometry and true pupil size.
        
        This method implements Stage 1 of the foreshortening correction algorithm,
        which jointly estimates the camera position (theta, phi) and the temporal
        dynamics of true pupil size A0(t) using B-splines. The data is automatically
        preprocessed (filtered and downsampled) before fitting.
        
        Parameters
        ----------
        eye : str
            Which eye to fit ('left' or 'right')
        r : float, str, or pint.Quantity, optional
            Eye-to-camera distance. If not provided, uses the 
            `r` from experimental_setup.
            - Plain number: assumed to be mm (with warning)
            - String: e.g., "600 mm", "60 cm"
            - Quantity: e.g., 600 * ureg.mm
        d : float, str, or pint.Quantity, optional
            Eye-to-screen distance. If not provided, uses the 
            `d` from experimental_setup.
            - Plain number: assumed to be mm (with warning)
            - String: e.g., "700 mm", "70 cm"
            - Quantity: e.g., 700 * ureg.mm
        intervals : Intervals, optional
            Time intervals to use for fitting. If None, uses all available data.
            Useful for fitting only on calibration periods.
        target_fs : float, default 50.0
            Target sampling rate in Hz after downsampling for efficiency.
            Pupil dynamics are typically well-preserved at 50 Hz.
        lowpass_freq : float, default 4.0
            Lowpass filter cutoff frequency in Hz. Removes high-frequency noise
            before downsampling. Should be below target_fs/2 (Nyquist).
        lambda_smooth : float, default 1.0
            Smoothness regularization weight. Higher values produce smoother
            A0(t) estimates. Tune via cross-validation if needed.
        knots_per_second : float, default 1.0
            Number of B-spline knots per second. Controls temporal resolution
            of A0(t) model. Typical values: 0.5-2.0.
        initial_theta : float, str, or pint.Quantity, optional
            Initial guess for camera polar angle. If None, uses
            pi/2 (horizontal) as starting point.
            - Plain number: assumed to be radians (with warning)
            - String: e.g., "20 degrees", "0.349 radians"
            - Quantity: e.g., 20 * ureg.degree
        initial_phi : float, str, or pint.Quantity, optional
            Initial guess for camera azimuthal angle. If None,
            uses 0 (camera on x-axis) as starting point.
            - Plain number: assumed to be radians (with warning)
            - String: e.g., "-90 degrees", "-1.57 radians"
            - Quantity: e.g., -90 * ureg.degree
        verbose : bool, default True
            Print optimization progress and fit statistics.
        
        Returns
        -------
        calibration : ForeshorteningCalibration
            Fitted calibration object containing camera geometry (theta, phi),
            spline coefficients for A0(t), and fit quality metrics.
        
        Raises
        ------
        ValueError
            If specified eye doesn't exist or lacks gaze/pupil data.
        
        Notes
        -----
        The model is:
            A_measured(x,y,t) = A0(t) * cos(alpha(x,y; theta, phi))
        
        where:
        - A0(t) = sum_k a_k * B_k(t) is modeled with B-splines
        - cos(alpha) is the foreshortening factor depending on viewing angle
        - theta, phi parameterize camera position in spherical coordinates
        
        The optimization minimizes:
            sum_i (A_i - A_pred_i)^2 + lambda_smooth * sum_k (a_{k+1} - a_k)^2
        
        Examples
        --------
        >>> import pypillometry as pp
        >>> data = pp.EyeData.from_eyelink('recording.edf')
        >>> 
        >>> # Set experimental setup
        >>> data.set_experimental_setup(  # doctest: +SKIP
        ...     screen_resolution=(1920, 1080),
        ...     physical_screen_size=("52 cm", "29 cm"),
        ...     eye_to_screen_perpendicular="65 cm",
        ...     camera_offset=("0 cm", "-30 cm", "0 cm"),
        ... )
        >>> 
        >>> # Fit on calibration period (first 2 minutes)
        >>> cal_intervals = data.get_intervals(end_time=120000)  # doctest: +SKIP
        >>> calib = data.fit_foreshortening(  # doctest: +SKIP
        ...     eye='left',
        ...     intervals=cal_intervals
        ... )
        >>> 
        >>> # Or provide r and d explicitly
        >>> calib = data.fit_foreshortening(  # doctest: +SKIP
        ...     eye='left',
        ...     r="60 cm",
        ...     d="70 cm",
        ...     intervals=cal_intervals
        ... )
        >>> 
        >>> # Inspect fit quality
        >>> print(calib)  # doctest: +SKIP
        >>> print(f"R² = {calib.fit_metrics['r2']:.3f}")  # doctest: +SKIP
        >>> 
        >>> # Compute correction for all data
        >>> correction_factor = calib.get_correction_factor(  # doctest: +SKIP
        ...     data['left', 'x'], 
        ...     data['left', 'y']
        ... )
        >>> corrected_pupil = data['left', 'pupil'] * correction_factor  # doctest: +SKIP
        
        See Also
        --------
        ForeshorteningCalibration : Container for fitted parameters
        correct_pupil_foreshortening : Simple foreshortening correction method
        """
        # Validate eye
        if eye not in self.eyes:
            raise ValueError(f"Eye '{eye}' not found in dataset. Available: {self.eyes}")
        
        # Check for required data
        x_var = f"{eye}_x"
        y_var = f"{eye}_y"
        pupil_var = f"{eye}_pupil"
        
        if x_var not in self.data or y_var not in self.data:
            raise ValueError(f"Gaze data (x, y) not available for {eye} eye")
        if pupil_var not in self.data:
            raise ValueError(f"Pupil data not available for {eye} eye")
        
        # Require experimental_setup for screen dimensions
        if self.experimental_setup is None:
            raise ValueError(
                "Cannot fit foreshortening: experimental_setup not set. "
                "Use set_experimental_setup() to configure it."
            )
        setup = self.experimental_setup
        
        # Get geometric parameters from experimental_setup or explicit args
        if r is None:
            if not setup.has_camera_position():
                raise ValueError(
                    "Camera position not set in experimental_setup. "
                    "Either provide r explicitly or set camera_offset in experimental_setup."
                )
            r = setup.r
            if verbose:
                logger.info(f"Using camera distance from experimental_setup: {r:.1f} mm")
        else:
            r = parse_distance(r)
        
        if d is None:
            if not setup.has_eye_distance():
                raise ValueError(
                    "Eye-to-screen distance not set in experimental_setup. "
                    "Either provide d explicitly or set eye_to_screen_perpendicular in experimental_setup."
                )
            d = setup.d
            if verbose:
                logger.info(f"Using screen distance from experimental_setup: {d:.1f} mm")
        else:
            d = parse_distance(d)
        
        # Parse angle parameters if provided
        if initial_theta is not None:
            initial_theta = parse_angle(initial_theta)
        
        if initial_phi is not None:
            initial_phi = parse_angle(initial_phi)
        
        # Validate geometric parameters
        if r <= 0:
            raise ValueError(f"Eye-to-camera distance r must be positive, got {r}")
        if d <= 0:
            raise ValueError(f"Eye-to-screen distance d must be positive, got {d}")
        
        if verbose:
            logger.info(f"Eye-to-camera distance: {r} mm")
            logger.info(f"Eye-to-screen distance: {d} mm")
        
        # Extract data (using masked array access to exclude blinks/artifacts)
        time = self["time_ms"].copy()
        x_gaze = self[eye, 'x'].copy()
        y_gaze = self[eye, 'y'].copy()
        pupil = self[eye, 'pupil'].copy()
        
        # Apply interval mask if provided
        if intervals is not None:
            mask = np.zeros(len(time), dtype=bool)
            for start, end in intervals:
                mask |= (time >= start) & (time <= end)
            
            time = time[mask]
            x_gaze = x_gaze[mask]
            y_gaze = y_gaze[mask]
            pupil = pupil[mask]
            
            if verbose:
                logger.info(f"Using {np.sum(mask)} samples from specified intervals")
        
        # Scale and center using experimental_setup
        scale_x = setup.mm_per_pixel_x
        scale_y = setup.mm_per_pixel_y
        
        x_gaze_mm = (x_gaze - setup.screen_width / 2) * scale_x
        y_gaze_mm = (y_gaze - setup.screen_height / 2) * scale_y
        
        # Remove invalid samples (NaN, masked, etc.)
        valid = np.isfinite(x_gaze_mm) & np.isfinite(y_gaze_mm) & np.isfinite(pupil) & (pupil > 0)
        
        time = time[valid]
        x_gaze_mm = x_gaze_mm[valid]
        y_gaze_mm = y_gaze_mm[valid]
        pupil = pupil[valid]
        
        n_valid = len(time)
        if n_valid < 100:
            raise ValueError(f"Insufficient valid samples for fitting: {n_valid} < 100")
        
        if verbose:
            logger.info(f"Valid samples after preprocessing: {n_valid}")
        
        # Preprocessing: lowpass filter
        if verbose:
            logger.info(f"Applying {lowpass_freq} Hz lowpass filter")
        
        nyquist = self.fs / 2
        if lowpass_freq >= nyquist:
            raise ValueError(f"Lowpass frequency ({lowpass_freq} Hz) must be < Nyquist ({nyquist} Hz)")
        
        # Apply Butterworth lowpass filter (order 3)
        pupil_filt = butter_lowpass_filter(pupil, cutoff=lowpass_freq, fs=self.fs, order=3)
        
        # Downsample
        if self.fs > target_fs:
            downsample_factor = int(np.round(self.fs / target_fs))
            
            if verbose:
                logger.info(f"Downsampling by factor {downsample_factor}: {self.fs} Hz → {self.fs/downsample_factor:.1f} Hz")
            
            time = downsample(time, downsample_factor)
            x_gaze_mm = downsample(x_gaze_mm, downsample_factor)
            y_gaze_mm = downsample(y_gaze_mm, downsample_factor)
            pupil_filt = downsample(pupil_filt, downsample_factor)
        
        n_samples = len(time)
        
        # Setup B-spline basis
        t_min, t_max = time[0], time[-1]
        duration_sec = (t_max - t_min) / 1000.0
        
        knots = _determine_knots(t_min, t_max, knots_per_second)
        basis_matrix = _create_bspline_basis(time, knots, degree=3)
        n_basis = basis_matrix.shape[1]
        
        if verbose:
            logger.info(f"B-spline basis: {n_basis} functions over {duration_sec:.1f} seconds")
        
        # Initialize parameters
        # Spline coefficients: initialize to mean pupil size
        a_init = np.full(n_basis, np.mean(pupil_filt))
        
        # Camera angles: use provided or default
        theta_init = initial_theta if initial_theta is not None else np.pi / 2  # Horizontal
        phi_init = initial_phi if initial_phi is not None else 0.0  # x-axis
        
        params_init = np.concatenate([a_init, [theta_init, phi_init]])
        
        # Setup bounds for L-BFGS-B
        bounds = [(0.1 * np.mean(pupil_filt), 10 * np.mean(pupil_filt))] * n_basis  # Spline coeffs > 0
        bounds += [(0, np.pi/2)]  # theta: polar angle from z-axis, camera must be forward of eye (0=on-axis, π/2=perpendicular)
        bounds += [(-np.pi, np.pi)]  # phi: azimuthal in x-y plane (-π/2=below, 0=right, π/2=above, ±π=left)
        
        # check that the initial values are within the bounds
        for i, (lb, ub) in enumerate(bounds):
            if params_init[i] < lb or params_init[i] > ub:
                logger.warning(f"Initial parameter {i} is out of bounds: {params_init[i]} not in [{lb}, {ub}]")
                params_init[i] = (lb + ub) / 2

        # Run optimization
        if verbose:
            logger.info("Starting optimization...")
        
        result = optimize.minimize(
            fun=_objective_function,
            x0=params_init,
            args=(x_gaze_mm, y_gaze_mm, time, pupil_filt, basis_matrix, r, d, lambda_smooth, 0.0),
            method='L-BFGS-B',
            bounds=bounds,
            jac=_objective_gradient,
            options={'disp': verbose, 'maxiter': 500}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        elif verbose:
            logger.info(f"Optimization converged in {result.nit} iterations")
        
        # Extract fitted parameters
        params_opt = result.x
        spline_coeffs_opt, theta_opt, phi_opt = _split_params(params_opt, n_basis)
        
        # Compute fit metrics
        from .foreshortening_calibration import _compute_cos_alpha_vectorized
        
        A0_fit = basis_matrix @ spline_coeffs_opt
        cos_alpha_fit = _compute_cos_alpha_vectorized(x_gaze_mm, y_gaze_mm, theta_opt, phi_opt, r, d, 0.0)
        pupil_pred = A0_fit * cos_alpha_fit
        
        residuals = pupil_filt - pupil_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((pupil_filt - np.mean(pupil_filt)) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        fit_metrics = {
            'r2': r2,
            'rmse': rmse,
            'n_samples': n_samples,
            'loss': result.fun,
            'n_iterations': result.nit,
            'converged': result.success
        }
        
        if verbose:
            logger.info(f"Fit quality: R² = {r2:.4f}, RMSE = {rmse:.4f}")
            logger.info(f"Camera position: theta = {np.degrees(theta_opt):.1f}°, phi = {np.degrees(phi_opt):.1f}°")
        
        # Create ForeshorteningCalibration object
        # Pass screen info from experimental_setup
        calibration = ForeshorteningCalibration(
            eye=eye,
            theta=theta_opt,
            phi=phi_opt,
            r=r,
            d=d,
            spline_coeffs=spline_coeffs_opt,
            spline_knots=knots,
            spline_degree=3,
            screen_resolution=setup.screen_resolution,
            physical_screen_size=setup.physical_screen_size,
            fit_intervals=intervals,
            fit_metrics=fit_metrics
        )
        
        return calibration
