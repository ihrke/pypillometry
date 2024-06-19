from collections.abc import MutableMapping, Iterable

def keys_to_string(key):
    if isinstance(key, str):
        key = [key]
    elif not isinstance(key, Iterable):
        raise ValueError("Key must be a string or iterable of strings")
    key=sorted(list(key))
    for k in key:
        if not isinstance(k, str):
            raise ValueError("Keys must be strings")
    key=",".join(key)
    return key


class Parameters(MutableMapping):
    """
    A dictionary that allows indexing by multiple keys in arbitrary order
    to store parameters. For example, a parameter "mean" for the right pupil
    can be saved as p["mean","right","pupil"]=10. Access is independent of
    order, so p["right","mean","pupil"] will return the same value.

    In addition, the dictionary will return a subset of itself if a key is 
    given that matches some of the stored keys. For example, p["right"] will
    return a `Parameters()` dictionary with all keys that contain a "right" 
    (with "right" removed from the key).

    In case a key is not found, a default value is returned. 

    """
    default_value: float
    data: dict

    def __init__(self, *args, default_value=None, **kwargs):
        """
        Initialize the Parameters object. Behaves like an ordinary 
        Python dictionary.

        Parameters
        ----------
        default_value: float
            default value to return in case a key is not found
        """
        self.data = dict()
        self.default_value = default_value
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def has_key(self,key, *args):
        """
        Check if a key is present in the dictionary. It only checks for 
        an exact match, i.e., there is a value for the key and not a subset.
        """
        if len(args)>0 and isinstance(key, str):
            key=(key,)+args
        key=keys_to_string(key)
        return key in self.data.keys()

    def __setitem__(self, key, value):
        key=keys_to_string(key)
        self.data[key]=value

    def __getitem__(self, key):
        key=keys_to_string(key)
        matches=[set(key.split(",")).issubset(set(k.split(","))) for k in self.data.keys()]
        if(sum(matches)==0): # no match -> return default
            return self.default_value
        elif sum(matches)==1: # exact match -> return parameter
            return self.data[key]
        else: # multiple matches -> return subset as Parameters dict
            r = Parameters(default_value=self.default_value)
            for k,v in self.data.items():
                nkey = tuple(set(k.split(",")).difference(set(key.split(","))))
                if set(key.split(",")).issubset(set(k.split(","))):
                    r[nkey]=v
            return r

    def __delitem__(self, key):
        key=keys_to_string(key)
        del self.data[key]

    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return repr(self.data)
