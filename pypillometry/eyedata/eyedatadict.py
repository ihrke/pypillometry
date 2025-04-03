from collections.abc import MutableMapping
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from numpy.typing import NDArray

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
        variables=[k.split("_")[1] for k in self.data.keys()]
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

    def __getitem__(self, key):
        # check if key is a tuple, in that case convert to string
        if isinstance(key, tuple):
            key="_".join(key)
        return self.data[key]


    def __setitem__(self, key: str, value: NDArray) -> None:
        if value is None or len(value) == 0:
            return
        value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Value must be numpy.ndarray, got {type(value)}")
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
        # check if key is a tuple, in that case convert to string
        if isinstance(key, tuple):
            key="_".join(key)
        self.data[key] = value.astype(float)
        self.mask[key] = np.zeros(self.shape, dtype=int)

    def __delitem__(self, key):
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
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Tuple key must have exactly 2 elements (eye, variable)")
            key = "_".join(key)
        if not isinstance(key, str):
            raise TypeError("Key must be string or tuple")
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

    def get_mask(self, key: str) -> NDArray:
        """Get mask for a specific key."""
        return self.mask[key]

    @property
    def variables(self) -> List[str]:
        """List of all available variables."""
        return self.get_available_variables()

    @property
    def eyes(self) -> List[str]:
        """List of all available eyes."""
        return self.get_available_eyes()
