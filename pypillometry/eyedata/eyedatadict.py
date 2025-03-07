from collections.abc import MutableMapping
import numpy as np

class EyeDataDict(MutableMapping):
    """
    A dictionary that contains 1-dimensional ndarrays of equal length
    and with the same datatype (float).
    Drops empty entries (None or length-0 ndarrays).

    Keys stored in this dictionary have the following shape:
    (eye)_(variable) where eye is either "left" or "right" or a 
    statistical combination (e.g., "mean" or "median" or "regress") and
    variable is one of "x", "y", "pupil" (or other calculated entities,
    for example "baseline" or "response" for the pupil).

    The dictionary can be indexed by a string "eye_variable" or by a tuple
    ("eye", "variable") like data["left","x"] or data["left_x"].
    """

    def __init__(self, *args, **kwargs):
        self.data = dict()
        self.mask = dict() # mask for missing/artifactual values
        self.length=0
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

    def get_eye(self, eye):
        """
        Return a subset EyeDataDict with all variables for a given eye.
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

    def __setitem__(self, key, value):
        if value is None or len(value)==0:
            return
        value=np.array(value)
        if not isinstance(value, np.ndarray):
            raise ValueError("Value must be numpy.ndarray")
        if len(value.shape)>1:
            raise ValueError("Array must be 1-dimensional")
        if self.length==0:
            self.length=value.shape[0]
        if value.shape[0]!=self.length:
            raise ValueError("Array must have same length as existing arrays")
        # check if key is a tuple, in that case convert to string
        if isinstance(key, tuple):
            key="_".join(key)
        self.data[key] = value.astype(float)
        self.mask[key] = np.zeros(self.length, dtype=int)

    def __delitem__(self, key):
        del self.data[key]
        del self.mask[key]

    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return self.length
    
    def __repr__(self) -> str:
        r="EyeDataDict(vars=%i,n=%i): \n"%(len(self.data), self.length)
        for k,v in self.data.items():
            r+="  %s (%s): "%(k,v.dtype)
            r+=", ".join(v[0:(min(5,self.length))].astype(str))
            if self.length>5:
                r+="..."
            r+="\n"
        return r
