from collections.abc import MutableMapping
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from numpy.typing import NDArray
import os
import tempfile
import h5py
from typing import Any
from loguru import logger
from ..convenience import ByteSize

class EyeDataDict(MutableMapping):
    """
    A dictionary that contains 1-dimensional ndarrays of equal length
    and with the same datatype (float).
    Drops empty entries (None or length-0 ndarrays).

    Keys stored in this dictionary have the following shape:
    (eye)_(variable) where eye can be any string identifier (e.g., "left", "right", 
    "mean", "median", "regress", or any other custom identifier) and variable is any 
    string identifier (e.g., "x", "y", "pupil", "baseline", "response", or any other 
    custom identifier).

    The dictionary can be indexed by a string "eye_variable" or by a tuple
    ("eye", "variable") like data["left","x"] or data["left_x"].
    """

    def __init__(self, *args, **kwargs) -> None:
        self.data: Dict[str, NDArray] = dict()
        self.mask: Dict[str, NDArray] = dict() # mask for missing/artifactual values
        self.length: int = 0
        self.shape: Optional[Tuple[int, ...]] = None
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def get_available_eyes(self, variable=None):
        """
        Return a list of available eyes.
        
        Parameters
        ----------
        variable : str, optional
            If specified, return only eyes for this variable.
        """
        if variable is not None:
            eyes=[k.split("_")[0] for k in self.data.keys() if k.endswith("_"+variable)]
        else:
            eyes=[k.split("_")[0] for k in self.data.keys()]
        return list(set(eyes))

    def get_available_variables(self):
        """
        Return a list of available variables.
        """
        variables = []
        for k in self.data.keys():
            parts = k.split("_", 1)  # Split only on first underscore
            if len(parts) > 1:
                variables.append(parts[1])
        return list(set(variables))

    def get_eye(self, eye: str) -> 'EyeDataDict':
        """
        Return a subset EyeDataDict with all variables for a given eye.
        
        Parameters
        ----------
        eye : str
            The eye to get data for ('left', 'right', 'mean', etc.)
        
        Returns
        -------
        EyeDataDict
            A new EyeDataDict containing only data for the specified eye
        
        Examples
        --------
        >>> d = EyeDataDict(left_x=[1,2], left_y=[3,4], right_x=[5,6])
        >>> left_data = d.get_eye('left')
        >>> print(left_data.data.keys())
        dict_keys(['left_x', 'left_y'])
        """
        return EyeDataDict({k:v for k,v in self.data.items() if k.startswith(eye+"_")})
    
    def get_variable(self, variable):
        """
        Return a subset EyeDataDict with all eyes for a given variable.
        """
        return EyeDataDict({k:v for k,v in self.data.items() if k.endswith("_"+variable)})

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = "_".join(key)
        return self.data[key]

    def __setitem__(self, key: str, value: NDArray) -> None:
        if value is None or len(value) == 0:
            return
        value = np.array(value).astype(float)
        if self.length > 0 and self.shape is not None:
            if value.shape != self.shape:
                raise ValueError(
                    f"Array must have shape {self.shape}, got {value.shape}"
                )
        if self.length==0 or self.shape is None:
            self.length=value.shape[0]
            self.shape=value.shape
        if np.any(np.array(self.shape)!=np.array(value.shape)):
            raise ValueError("Array must have same dimensions as existing arrays")
        key = self._validate_key(key)  # Only validate key when setting values
        
        # For pupil variables, convert 0 values to NaN (0 pupil size is invalid)
        if 'pupil' in key:
            value = np.where(value == 0, np.nan, value)
        
        self.data[key] = value
        self.mask[key] = np.zeros(self.shape, dtype=int)
        self.mask[key][np.isnan(value)] = 1 # set mask to 1 for missing values

    def set_with_mask(self, key: Union[str, Tuple[str, str]], value: NDArray, 
                      mask: Optional[NDArray] = None, preserve_mask: bool = False) -> None:
        """
        Set data array with explicit mask control.
        
        This method provides more control over mask handling than __setitem__,
        which always resets the mask to zeros (except for NaN values).
        
        Parameters
        ----------
        key : str or tuple
            Dictionary key for the data (e.g., "left_pupil" or ("left", "pupil"))
        value : NDArray
            Data array to store
        mask : NDArray, optional
            Mask array to use. If None and preserve_mask=True, keeps existing mask.
            If None and preserve_mask=False, creates new zero mask (with NaNs masked).
        preserve_mask : bool, default False
            If True and mask is None, preserve the existing mask for this key.
            Ignored if mask is explicitly provided.
        
        Examples
        --------
        >>> # Set data and preserve existing mask
        >>> data_dict.set_with_mask("left_pupil", new_values, preserve_mask=True)
        
        >>> # Set data with explicit mask
        >>> data_dict.set_with_mask("left_pupil", new_values, mask=new_mask)
        
        >>> # Set data with fresh zero mask (same as __setitem__)
        >>> data_dict.set_with_mask("left_pupil", new_values)
        """
        if value is None or len(value) == 0:
            return
        
        value = np.array(value)
        
        # Shape validation
        if self.length > 0 and self.shape is not None:
            if value.shape != self.shape:
                raise ValueError(
                    f"Array must have shape {self.shape}, got {value.shape}"
                )
        if self.length == 0 or self.shape is None:
            self.length = value.shape[0]
            self.shape = value.shape
        if np.any(np.array(self.shape) != np.array(value.shape)):
            raise ValueError("Array must have same dimensions as existing arrays")
        
        key = self._validate_key(key)
        
        # For pupil variables, convert 0 values to NaN (0 pupil size is invalid)
        value_float = value.astype(float)
        if 'pupil' in key:
            value_float = np.where(value_float == 0, np.nan, value_float)
        
        # Store data
        self.data[key] = value_float
        
        # Handle mask
        if mask is not None:
            # Explicit mask provided
            self.mask[key] = np.asarray(mask).astype(int)
        elif preserve_mask and key in self.mask:
            # Preserve existing mask - do nothing
            pass
        else:
            # Create new mask (default behavior, same as __setitem__)
            self.mask[key] = np.zeros(self.shape, dtype=int)
            self.mask[key][np.isnan(value_float)] = 1

    def __delitem__(self, key):
        if isinstance(key, tuple):
            key = "_".join(key)
        del self.data[key]
        del self.mask[key]

    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return self.length
    
    def __repr__(self) -> str:
        r="EyeDataDict(vars=%i,n=%i,shape=%s): \n"%(len(self.data), self.length, str(self.shape))
        for k,v in self.data.items():
            r+="  %s (%s): "%(k,v.dtype)
            r+=", ".join(v.flat[0:(min(5,self.length))].astype(str).tolist())
            if self.length>5:
                r+="..."
            r+="\n"
        return r

    def _validate_key(self, key: Union[str, Tuple[str, str]]) -> str:
        """Validate and normalize key format."""
        if not isinstance(key, (str, tuple)):
            raise TypeError("Key must be string or tuple")
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Tuple key must have exactly 2 elements (eye, variable)")
            key = "_".join(key)
        if "_" not in key:
            raise ValueError("Key must be in format 'eye_variable'")
        return key

    def copy(self) -> 'EyeDataDict':
        """Create a deep copy of the dictionary."""
        new_dict = EyeDataDict()
        for key, value in self.data.items():
            new_dict[key] = value.copy()
        return new_dict

    def set_mask(self, key: str, mask: NDArray) -> None:
        """Set mask for a specific key."""
        if key not in self.data:
            raise KeyError(f"No data for key {key}")
        if mask.shape != self.shape:
            raise ValueError(f"Mask must have shape {self.shape}")
        self.mask[key] = mask.astype(int)

    def get_mask(self, key: str = None) -> NDArray:
        """Get mask for a specific key or joint mask across all keys.
        
        Parameters
        ----------
        key : str, optional
            Key to get mask for. If None, returns logical OR of all masks.
            
        Returns
        -------
        NDArray
            Mask array for specified key, or joint mask if no key given.
        """
        if key is not None:
            return self.mask[key]
        
        # Get joint mask across all keys
        joint_mask = np.zeros(self.shape, dtype=int)
        for mask in self.mask.values():
            joint_mask = np.logical_or(joint_mask, mask)
        return joint_mask

    @property
    def variables(self) -> List[str]:
        """List of all available variables."""
        return self.get_available_variables()

    @property
    def eyes(self) -> List[str]:
        """List of all available eyes."""
        return self.get_available_eyes()

    def get_size(self) -> ByteSize:
        """Return the size of the dictionary in bytes.
        
        Returns
        -------
        ByteSize
            Total size in bytes.
        """
        total_size = 0
        for arr in self.data.values():
            total_size += arr.nbytes
        for arr in self.mask.values():
            total_size += arr.nbytes
        return ByteSize(total_size)

    def count_masked(self, per_key: bool = False) -> Union[int, Dict[str, int]]:
        """Count the number of missing values based on masks.
        
        Parameters
        ----------
        per_key : bool, optional
            If True, return a dictionary with counts per key.
            If False, return maximum count across all keys.
            Default is False.
            
        Returns
        -------
        Union[int, Dict[str, int]]
            If per_key is False, returns maximum number of missing values across all keys.
            If per_key is True, returns a dictionary mapping keys to their missing value counts.
            
        Examples
        --------
        >>> d = EyeDataDict(left_x=[1,2,3], left_y=[4,5,6])
        >>> d.set_mask("left_x", [0,1,0])  # second value is missing
        >>> d.set_mask("left_y", [1,0,1])  # first and last values are missing
        >>> d.count_masked()  # max missing values
        2
        >>> d.count_masked(per_key=True)  # missing values per key
        {'left_x': 1, 'left_y': 2}
        """
        if per_key:
            return {key: np.sum(mask) for key, mask in self.mask.items()}
        else:
            return max(np.sum(mask) for mask in self.mask.values())

