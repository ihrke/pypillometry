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

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = "_".join(key)
        return self.data[key]

    def __setitem__(self, key: str, value: NDArray) -> None:
        if value is None or len(value) == 0:
            return
        value = np.array(value)
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
        self.data[key] = value.astype(float)
        self.mask[key] = np.zeros(self.shape, dtype=int)
        self.mask[key][np.isnan(value)] = 1 # set mask to 1 for missing values

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
        
class CachedEyeDataDict(EyeDataDict):
    def __init__(self, *args, cache_dir: Optional[str] = None, max_memory_mb: float = 100, **kwargs):
        """Initialize a cached version of EyeDataDict.
        
        Parameters
        ----------
        cache_dir : str, optional
            Directory to store cache files. If None, creates a temporary directory.
        max_memory_mb : float, optional
            Maximum memory usage in MB. Default is 100MB.
        """
        # Initialize base class without data
        self.data: Dict[str, np.ndarray] = {}
        self.mask: Dict[str, np.ndarray] = {}
        self.length: int = 0
        self.shape: Optional[tuple] = None
        
        # Cache settings
        self._cache_dir = cache_dir or tempfile.mkdtemp()
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Created cache directory at {cache_dir}")
            
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_memory_bytes = 0
        
        # Initialize single HDF5 file
        self._init_h5_file()
        
        # Track memory usage and access patterns
        self._in_memory_data: Dict[str, np.ndarray] = {}
        self._in_memory_mask: Dict[str, np.ndarray] = {}
        self._array_sizes: Dict[str, int] = {}
        self._access_counts: Dict[str, int] = {}
        
        # Add any initial data
        if args or kwargs:
            self.update(dict(*args, **kwargs))

    def _init_h5_file(self):
        """Initialize single HDF5 file with data and mask groups."""
        if not hasattr(self, '_h5_file'):
            self._h5_path = os.path.join(self._cache_dir, 'eyedata_cache.h5')
            self._h5_file = h5py.File(self._h5_path, 'a')
            
            # Create groups if they don't exist
            if 'data' not in self._h5_file:
                self._h5_file.create_group('data')
            if 'mask' not in self._h5_file:
                self._h5_file.create_group('mask')

    def _get_array_size(self, arr: np.ndarray) -> int:
        """Calculate size of numpy array in bytes."""
        return arr.nbytes

    def _update_cache(self, key: str, data: np.ndarray, mask: np.ndarray):
        """Update memory cache using LRU strategy with size limits."""
        total_size = self._get_array_size(data) + self._get_array_size(mask)
        
        # If arrays are too large to fit in cache, don't cache them
        if total_size > self._max_memory_bytes:
            return
            
        # Remove least recently used arrays until we have enough space
        while (self._current_memory_bytes + total_size > self._max_memory_bytes and 
               self._in_memory_data):
            # Find least recently used key
            lru_key = min(self._access_counts.items(), key=lambda x: x[1])[0]
            self._current_memory_bytes -= self._array_sizes[lru_key]
            del self._in_memory_data[lru_key]
            del self._in_memory_mask[lru_key]
            del self._array_sizes[lru_key]
            del self._access_counts[lru_key]
            
        # Add new arrays to cache
        self._in_memory_data[key] = data
        self._in_memory_mask[key] = mask
        self._array_sizes[key] = total_size
        self._current_memory_bytes += total_size
        self._access_counts[key] = max(self._access_counts.values(), default=0) + 1

    def __setitem__(self, key: str, value: np.ndarray):
        """Set item in cache and HDF5."""
        if value is None or len(value) == 0:
            return
            
        value = np.array(value)
        if self.length > 0 and self.shape is not None:
            if value.shape != self.shape:
                raise ValueError(f"Array must have shape {self.shape}, got {value.shape}")
        if self.length == 0 or self.shape is None:
            self.length = value.shape[0]
            self.shape = value.shape
        if np.any(np.array(self.shape) != np.array(value.shape)):
            raise ValueError("Array must have same dimensions as existing arrays")
            
        key = self._validate_key(key)
        value = value.astype(float)
        mask = np.zeros(self.shape, dtype=int)
        
        # Store in HDF5
        if key in self._h5_file['data']:
            del self._h5_file['data'][key]
        if key in self._h5_file['mask']:
            del self._h5_file['mask'][key]
        self._h5_file['data'].create_dataset(key, data=value)
        self._h5_file['mask'].create_dataset(key, data=mask)
        self._h5_file.flush()
        
        # Update memory cache
        self._update_cache(key, value, mask)
        
        # Update base class data
        self.data[key] = value
        self.mask[key] = mask
        
        # Initialize access count for new key
        self._access_counts[key] = max(self._access_counts.values(), default=0) + 1

    def __getitem__(self, key: str) -> np.ndarray:
        """Get item from cache or disk."""
        key = self._validate_key(key)
        
        # Update access count for LRU
        if key in self._access_counts:
            self._access_counts[key] = max(self._access_counts.values()) + 1
        
        # Try to get from memory cache first
        if key in self._in_memory_data:
            return self._in_memory_data[key]
            
        # If not in memory, try to get from HDF5
        if key in self._h5_file['data']:
            data = self._h5_file['data'][key][:]
            mask = self._h5_file['mask'][key][:] if key in self._h5_file['mask'] else np.zeros_like(data, dtype=int)
            
            # Try to cache the data in memory
            self._update_cache(key, data, mask)
            return data
            
        raise KeyError(key)

    def get_mask(self, key: Union[str, None] = None) -> np.ndarray:
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
            key = self._validate_key(key)
            # Try memory cache first
            if key in self._in_memory_mask:
                return self._in_memory_mask[key]
                
            # Load from HDF5
            if key in self._h5_file['mask']:
                mask = self._h5_file['mask'][key][:]
                if key in self._h5_file['data']:
                    data = self._h5_file['data'][key][:]
                    self._update_cache(key, data, mask)
                return mask
                
            raise KeyError(key)
        
        # Get joint mask by recursively calling get_mask for each key
        joint_mask = np.zeros(self.shape, dtype=int)
        for k in self.keys():
            mask = self.get_mask(k)  # This will handle both memory and HDF5 cases
            joint_mask = np.logical_or(joint_mask, mask)
        return joint_mask

    def set_mask(self, key: str, mask: np.ndarray):
        """Set mask for a specific key."""
        key = self._validate_key(key)
        mask = np.array(mask, dtype=int)
        
        # Store in HDF5
        if key in self._h5_file['mask']:
            del self._h5_file['mask'][key]
        self._h5_file['mask'].create_dataset(key, data=mask)
        
        # Update memory cache if key is cached
        if key in self._in_memory_data:
            self._update_cache(key, self._in_memory_data[key], mask)

    def set_max_memory(self, max_memory_mb: float):
        """Set maximum memory usage in MB."""
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        # If new limit is lower, remove excess arrays
        while self._current_memory_bytes > self._max_memory_bytes and self._in_memory_data:
            lru_key = min(self._access_counts.items(), key=lambda x: x[1])[0]
            self._current_memory_bytes -= self._array_sizes[lru_key]
            del self._in_memory_data[lru_key]
            del self._in_memory_mask[lru_key]
            del self._array_sizes[lru_key]
            del self._access_counts[lru_key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        # Calculate actual memory usage of in-memory arrays
        memory_used = sum(arr.nbytes for arr in self._in_memory_data.values())
        memory_used += sum(arr.nbytes for arr in self._in_memory_mask.values())
        
        return {
            'memory_used_mb': memory_used / (1024 * 1024),
            'memory_limit_mb': self._max_memory_bytes / (1024 * 1024),
            'arrays_in_memory': len(self._in_memory_data),
            'arrays_on_disk': len(self._h5_file['data']),
            'memory_usage_per_array': {
                k: (self._in_memory_data[k].nbytes + self._in_memory_mask[k].nbytes) / (1024 * 1024) 
                for k in self._in_memory_data.keys()
            }
        }

    def clear_cache(self):
        """Clear all cached data from memory and disk."""
        # Clear memory cache
        self._in_memory_data.clear()
        self._in_memory_mask.clear()
        self._array_sizes.clear()
        self._access_counts.clear()
        self._current_memory_bytes = 0
        
        # Clear HDF5 file
        if hasattr(self, '_h5_file'):
            del self._h5_file['data']
            del self._h5_file['mask']
            self._h5_file.create_group('data')
            self._h5_file.create_group('mask')
            self._h5_file.flush()
        
        # Clear base class data
        self.data.clear()
        self.mask.clear()
        self.length = 0
        self.shape = None

    def __del__(self):
        """Clean up HDF5 file."""
        if hasattr(self, '_h5_file'):
            self._h5_file.close()

    def get_size(self) -> ByteSize:
        """Return the size of the dictionary in bytes, split by storage location.
        
        Returns
        -------
        ByteSize
            Total size in bytes, with cached portion if applicable.
        """
        memory_size = 0
        for arr in self._in_memory_data.values():
            memory_size += arr.nbytes
        for arr in self._in_memory_mask.values():
            memory_size += arr.nbytes
            
        disk_size = 0
        for key in self._h5_file['data'].keys():
            if key not in self._in_memory_data:  # Only count arrays not in memory
                disk_size += self._h5_file['data'][key].nbytes
                if key in self._h5_file['mask']:
                    disk_size += self._h5_file['mask'][key].nbytes
                    
        return ByteSize({
            'memory': memory_size,
            'disk': disk_size
        })

    def __repr__(self) -> str:
        """Return a string representation of the cached dictionary."""
        r = "CachedEyeDataDict(vars=%i,n=%i,shape=%s): \n" % (len(self.data), self.length, str(self.shape))
        r += f"  Cache dir: {self._cache_dir}\n"
        r += f"  Memory limit: {self._max_memory_bytes / (1024*1024):.1f} MB\n"
        r += f"  Current memory: {self._current_memory_bytes / (1024*1024):.1f} MB\n"
        r += f"  Arrays in memory: {len(self._in_memory_data)}\n"
        r += f"  Arrays on disk: {len(self._h5_file['data'])}\n"
        for k, v in self.data.items():
            r += "  %s (%s): " % (k, v.dtype)
            r += ", ".join(v.flat[0:(min(5, self.length))].astype(str).tolist())
            if self.length > 5:
                r += "..."
            r += "\n"
        return r
