"""Synthetic eye-tracking data for testing and validation."""

import inspect
from .eyedata import EyeData
from .eyedatadict import EyeDataDict


class FakeEyeData(EyeData):
    """
    EyeData with simulation metadata for testing and validation.
    
    Extends EyeData with four attributes:
    - sim_fct: Reference to the function used to generate this data
    - sim_fct_name: Name of the generation function (str)
    - sim_params: Dictionary with generation parameters (excludes function reference)
    - sim_data: EyeDataDict with ground truth time series
    
    This class enables:
    1. Printing how the data was generated via get_generation_call()
    2. Regenerating data with the same or modified parameters via regenerate()
    3. Accessing ground truth for validation
    """
    
    def __init__(self, 
                 time, left_x=None, left_y=None, left_pupil=None,
                 right_x=None, right_y=None, right_pupil=None,
                 sim_fct=None, sim_fct_name=None,
                 sim_params=None, sim_data=None,
                 **kwargs):
        """
        Initialize FakeEyeData.
        
        Parameters
        ----------
        time, left_*, right_* : arrays
            Standard EyeData parameters
        sim_fct : callable, optional
            Function reference used to generate this data
        sim_fct_name : str, optional
            Name of generation function (auto-detected from sim_fct if not provided)
        sim_params : dict, optional
            Parameters used for generation (excludes function reference)
        sim_data : EyeDataDict, optional
            Ground truth time series
        **kwargs : additional EyeData parameters
        """
        super().__init__(time=time, left_x=left_x, left_y=left_y, 
                        left_pupil=left_pupil, right_x=right_x, 
                        right_y=right_y, right_pupil=right_pupil, **kwargs)
        
        self.sim_fct = sim_fct
        self.sim_fct_name = sim_fct_name or (sim_fct.__name__ if sim_fct else None)
        self.sim_params = sim_params or {}
        self.sim_data = sim_data if sim_data is not None else EyeDataDict()
    
    def get_generation_call(self):
        """
        Return string representation of the function call used to generate this data.
        
        Returns
        -------
        str
            String like "generate_foreshortening_data(duration=60, fs=1000, ...)"
        
        Examples
        --------
        >>> data = generate_foreshortening_data(duration=60, seed=42)
        >>> print(data.get_generation_call())
        generate_foreshortening_data(duration=60, fs=1000, eye='left', ...)
        """
        if not self.sim_fct_name:
            return "Unknown generation function"
        
        param_strs = [f"{k}={repr(v)}" for k, v in self.sim_params.items()]
        return f"{self.sim_fct_name}({', '.join(param_strs)})"
    
    def regenerate(self, **override_params):
        """
        Regenerate data using stored function and parameters.
        
        Parameters
        ----------
        **override_params : dict
            Parameters to override from stored sim_params
        
        Returns
        -------
        FakeEyeData
            Newly generated data with same or modified parameters
        
        Raises
        ------
        ValueError
            If sim_fct is None
        
        Examples
        --------
        >>> data = generate_foreshortening_data(duration=60, seed=42)
        >>> data2 = data.regenerate(seed=43)  # Same params, different seed
        >>> data3 = data.regenerate(duration=120, seed=44)  # Different duration
        """
        if self.sim_fct is None:
            raise ValueError("Cannot regenerate: sim_fct is None")
        
        params = self.sim_params.copy()
        params.update(override_params)
        return self.sim_fct(**params)
    
    def __repr__(self):
        """Enhanced repr showing simulation info."""
        base_repr = super().__repr__()
        sim_keys = list(self.sim_data.keys()) if self.sim_data else []
        return f"Fake{base_repr[:-1]}, sim_fct={self.sim_fct_name}, sim_data={sim_keys})"

