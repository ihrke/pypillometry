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
        if len(args)>0 and isinstance(key, str):
            key=(key,)+args
        key=keys_to_string(key)
        return key in self.data.keys()

    def __setitem__(self, key, value):
        key=keys_to_string(key)
        self.data[key]=value

    def __getitem__(self, key):
        key=keys_to_string(key)
        if key not in self.data:
            return self.default_value
        else:
            return self.data[key]

    def __delitem__(self, key):
        key=keys_to_string(key)
        del self.data[key]

    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return repr(self.data)
