from collections.abc import Mapping, MutableMapping
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from numpy.typing import NDArray
import os
import tempfile
import h5py
from typing import Any
from loguru import logger
from ..convenience import ByteSize, _round_values
from ..state import Stateful


def _check_eye_name(eye: str) -> None:
    """Raise a ValueError if `eye` cannot be used as an eye name.

    Keys are ``"<eye>_<variable>"`` and the variable is everything after the
    first ``_``, so eye names may not contain ``_``.
    """
    if not isinstance(eye, str) or eye == "" or "_" in eye:
        raise ValueError(
            f"Invalid eye name {eye!r}: eye names must be non-empty and may not "
            "contain '_' (keys are '<eye>_<variable>' and the variable is "
            "everything after the first '_')")


def _join_key(key: Tuple[str, str]) -> str:
    """Join an ``(eye, variable)`` tuple to a key, checking the eye name."""
    if len(key) != 2:
        raise ValueError("Tuple key must have exactly 2 elements (eye, variable)")
    _check_eye_name(key[0])
    return "_".join(key)


#: dtype of the layer bit-fields (one bit per layer)
LAYER_DTYPE = np.uint16
#: maximum number of layers (bits of `LAYER_DTYPE`)
MAX_LAYERS = 16


#: the standard layers: missing data (blink, offscreen, divergence, user),
#: reconstructed data (interpolated) and periods outside the analysis (excluded)
STANDARD_LAYERS = ("blink", "interpolated", "offscreen", "divergence", "user", "excluded")
#: layers stored in `EyeDataDict.rec_layers` (recording-wide)
REC_LAYERS = ("excluded",)


class MaskPolicy:
    """
    Which samples count as unusable (masked).

    A sample is masked if its value is NaN (with `nan`), or if one of the
    `layers` is set and none of the layers in `unless` is set. Recording-wide
    layers (e.g., ``"excluded"``) mask a sample regardless of `unless`.

    The predefined policies are in `POLICIES`: ``"default"`` (interpolated
    samples are used), ``"strict"`` (interpolated samples are masked) and
    ``"raw"`` (nothing is masked).

    Parameters
    ----------
    layers : str or list of str
        layers that make a sample unusable
    unless : str or list of str
        layers that make a sample usable again (e.g., ``"interpolated"``:
        the value was reconstructed)
    nan : bool
        NaN values are unusable
    name : str, optional
        name of the policy

    Examples
    --------
    >>> d.mask("left", "pupil", policy=MaskPolicy(["blink"]))  # doctest: +SKIP
    """

    def __init__(self, layers=(), unless=(), nan: bool = True, name: Optional[str] = None):
        self.layers = (layers,) if isinstance(layers, str) else tuple(layers)
        self.unless = (unless,) if isinstance(unless, str) else tuple(unless)
        self.nan = bool(nan)
        self.name = name

    def __repr__(self):
        return (f"MaskPolicy(layers={list(self.layers)}, unless={list(self.unless)}, "
                f"nan={self.nan}" + (f", name={self.name!r})" if self.name else ")"))


# blinks mask the data where they were applied as mask (``apply_mask=True``,
# internal layer "_blink_applied")
_MISSING = ("_blink_applied", "offscreen", "divergence", "user", "excluded")
#: predefined policies (see :class:`MaskPolicy`)
POLICIES = {
    "default": MaskPolicy(_MISSING, unless="interpolated", name="default"),
    "strict": MaskPolicy(_MISSING + ("interpolated",), name="strict"),
    "raw": MaskPolicy(nan=False, name="raw"),
}


def _as_policy(policy) -> MaskPolicy:
    """MaskPolicy from a policy, the name of a predefined one or a list of layers."""
    if isinstance(policy, MaskPolicy):
        return policy
    if isinstance(policy, str):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; predefined policies: "
                             f"{', '.join(POLICIES)} (or a MaskPolicy or a list of layers)")
        return POLICIES[policy]
    return MaskPolicy(policy)


def _without_layer(policy, name: str) -> MaskPolicy:
    """`policy` without the layer `name` (e.g., the layer that a detector is
    about to replace)."""
    policy = _as_policy(policy)
    return MaskPolicy([n for n in policy.layers if n != name], policy.unless, policy.nan)


class _PolicyMask(Mapping):
    """
    Read-only mapping key -> int mask (1: masked) of an :class:`EyeDataDict`
    under the default mask policy (see `POLICIES`).

    The arrays are computed on access and are read-only; change masks with
    the methods of :class:`EyeDataDict` (e.g., `set_mask`, `add_mask`).
    """

    def __init__(self, d):
        self._d = d

    def __getitem__(self, key):
        out = self._d.get_policy_mask(key).astype(int)
        out.flags.writeable = False
        return out

    def __iter__(self):
        return iter(self._d.data)

    def __len__(self):
        return len(self._d.data)

    def __contains__(self, key):
        return key in self._d.data


class EyeDataDict(MutableMapping, Stateful):
    """
    A dictionary that contains 1-dimensional ndarrays of equal length
    and with the same datatype (float).
    Drops empty entries (None or length-0 ndarrays).

    Keys stored in this dictionary have the following shape:
    (eye)_(variable) where eye can be any string identifier (e.g., "left", "right", 
    "mean", "median", "regress", or any other custom identifier) and variable is any 
    string identifier (e.g., "x", "y", "pupil", "baseline", "response", or any other 
    custom identifier).

    Values are stored as given (converted to float); NaN values are masked.
    No other values are treated as missing (e.g., a pupil size of 0 is kept);
    converting missing-value codes to NaN is up to the code that reads the data.

    Each key also has mask *layers*: a bit-field (`layers`, dtype
    `LAYER_DTYPE`) with one bit per named layer (e.g., ``"blink"``). The
    layer names are registered in `layer_names` (the same for all keys);
    a layer gets the next free bit when it is first set. Layers that
    apply to all keys (e.g., ``"excluded"``) are stored once, in
    `rec_layers` (recording-wide layers, same registry). `mask` is the
    mask under the default mask policy as one 0/1 array per key (see
    :class:`MaskPolicy`).

    The dictionary can be indexed by a string "eye_variable" or by a tuple
    ("eye", "variable") like data["left","x"] or data["left_x"].
    """

    # state (see pypillometry.state): data, masks and layers
    _STATE_FIELDS = (("values", None, "data"), "layers", "layer_names", "rec_layers")
    _STATE_EXCLUDE = ("length", "shape")

    def __init__(self, *args, **kwargs) -> None:
        self.data: Dict[str, NDArray] = dict()
        self.layers: Dict[str, NDArray] = dict()  # layer bit-fields
        self.layer_names: List[str] = []  # layer i is bit 1 << i
        # recording-wide layers (one bit-field for all keys; None without data)
        self.rec_layers: Optional[NDArray] = None
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

        Returns
        -------
        list of str
            names of the available eyes (e.g., ``["left", "right"]``), in the
            order in which they were added
        """
        if variable is not None:
            eyes=[k.split("_")[0] for k in self.data.keys() if k.partition("_")[2] == variable]
        else:
            eyes=[k.split("_")[0] for k in self.data.keys()]
        return list(dict.fromkeys(eyes))

    def get_available_variables(self):
        """
        Return a list of available variables.

        Returns
        -------
        list of str
            names of the available variables (e.g., ``["x", "y", "pupil"]``),
            in the order in which they were added
        """
        variables = []
        for k in self.data.keys():
            parts = k.split("_", 1)  # Split only on first underscore
            if len(parts) > 1:
                variables.append(parts[1])
        return list(dict.fromkeys(variables))

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

        Parameters
        ----------
        variable : str
            The variable to get data for ('x', 'y', 'pupil', etc.)

        Returns
        -------
        EyeDataDict
            A new EyeDataDict containing only data for the specified variable
        """
        return EyeDataDict({k:v for k,v in self.data.items() if k.partition("_")[2] == variable})

    def select(self, eyes=None, variables=None) -> List[Tuple[str, str]]:
        """
        Return the existing (eye, variable) pairs for the given eyes and variables.

        Only pairs for which data is stored are returned, so data where not
        every eye has every variable (e.g., ``left_pupil``, ``left_x``,
        ``right_pupil``) can be processed pair by pair.

        Parameters
        ----------
        eyes : str or list of str, optional
            eye(s) to select; if None or empty, all available eyes
        variables : str or list of str, optional
            variable(s) to select; if None or empty, all available variables

        Returns
        -------
        list of tuple
            ``(eye, variable)`` pairs, ordered by eye, then variable (in the
            order given or, for all eyes/variables, the order of the data)

        Raises
        ------
        KeyError
            if something that was explicitly requested does not exist: a
            pair (if both `eyes` and `variables` are given), or an eye or
            variable without any data among the selected pairs (if only
            one of them is given). The message lists the available channels.
        ValueError
            if a requested eye name contains ``_`` (eye names may not
            contain ``_``, see :meth:`_validate_key`)

        Examples
        --------
        >>> d = EyeDataDict(left_pupil=[1,2], left_x=[3,4], right_pupil=[5,6])
        >>> d.select()
        [('left', 'pupil'), ('left', 'x'), ('right', 'pupil')]
        >>> d.select(variables="x")
        [('left', 'x')]
        """
        eyes_given = eyes is not None and len(eyes) > 0
        vars_given = variables is not None and len(variables) > 0
        eyes = ([eyes] if isinstance(eyes, str) else list(eyes)) if eyes_given else self.eyes
        variables = (([variables] if isinstance(variables, str) else list(variables))
                     if vars_given else self.variables)

        pairs = [(e, v) for e in eyes for v in variables if f"{e}_{v}" in self.data]
        if eyes_given:
            for eye in eyes:
                _check_eye_name(eye)
        if eyes_given and vars_given:
            missing = [f"{e}_{v}" for e in eyes for v in variables if (e, v) not in pairs]
        elif eyes_given:
            missing = [e for e in eyes if not any(pe == e for pe, _ in pairs)]
        elif vars_given:
            missing = [v for v in variables if not any(pv == v for _, pv in pairs)]
        else:
            missing = []
        if missing:
            raise KeyError("No data for %s; available channels: %s"
                           % (", ".join(missing), ", ".join(self.data.keys())))
        return pairs

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = _join_key(key)
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

        self.data[key] = value
        self.layers[key] = np.zeros(self.shape, dtype=LAYER_DTYPE)  # NaN: from the values
        self._init_rec_layers()

    def set_with_mask(self, key: Union[str, Tuple[str, str]], value: NDArray, 
                      mask: Optional[NDArray] = None, preserve_mask: bool = False) -> None:
        """
        Set data array with explicit mask control.
        
        This method provides more control over mask handling than __setitem__,
        which always resets the mask to zeros (except for NaN values) and
        clears the layers.
        
        Parameters
        ----------
        key : str or tuple
            Dictionary key for the data (e.g., "left_pupil" or ("left", "pupil"))
        value : NDArray
            Data array to store
        mask : NDArray, optional
            Mask array to use (see `set_mask`). If None and preserve_mask=True,
            keeps existing mask and layers. If None and preserve_mask=False,
            creates new zero mask (with NaNs masked) and clears the layers.
        preserve_mask : bool, default False
            If True and mask is None, preserve the existing mask and layers
            for this key. Ignored if mask is explicitly provided.
        
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
        
        value_float = value.astype(float)

        existed = key in self.data
        # Store data
        self.data[key] = value_float
        self._init_rec_layers()
        
        # Handle mask
        if mask is not None and existed:
            # Explicit mask provided
            self.set_mask(key, mask)
        elif mask is not None:
            self.layers[key] = np.zeros(self.shape, dtype=LAYER_DTYPE)
            if np.any(mask):
                self.set_layer(key, "user", mask)
        elif preserve_mask and existed:
            # Preserve existing mask - do nothing
            pass
        else:
            # Create new mask (default behavior, same as __setitem__)
            self.layers[key] = np.zeros(self.shape, dtype=LAYER_DTYPE)

    def _init_rec_layers(self) -> None:
        """Create empty recording-wide layers once the length is known."""
        if self.rec_layers is None:
            self.rec_layers = np.zeros(self.shape, dtype=LAYER_DTYPE)

    def _set_channel(self, key: str, values: NDArray, layers: NDArray) -> None:
        """Set values and layers of `key` as given (no other rules)."""
        self.set_with_mask(key, values, mask=np.zeros(np.shape(values), dtype=bool))
        self.layers[key] = np.asarray(layers, dtype=LAYER_DTYPE).copy()

    def __delitem__(self, key):
        if isinstance(key, tuple):
            key = _join_key(key)
        del self.data[key]
        del self.layers[key]

    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return self.length
    
    def __repr__(self) -> str:
        r="EyeDataDict(vars=%i,n=%i,shape=%s): \n"%(len(self.data), self.length, str(self.shape))
        for k,v in self.data.items():
            r+="  %s (%s): "%(k,v.dtype)
            r+=", ".join(str(_round_values(x)) for x in v.flat[0:(min(5,self.length))])
            if self.length>5:
                r+="..."
            r+="\n"
        return r

    def _validate_key(self, key: Union[str, Tuple[str, str]]) -> str:
        """Validate and normalize key format.

        Tuple keys ``(eye, variable)`` are joined with ``_``; the eye name may
        not contain ``_``, since the variable is everything after the first
        ``_`` (e.g., ``("my_eye", "pupil")`` would be read back as eye
        ``"my"``, variable ``"eye_pupil"``).
        """
        if not isinstance(key, (str, tuple)):
            raise TypeError("Key must be string or tuple")
        if isinstance(key, tuple):
            key = _join_key(key)
        if "_" not in key:
            raise ValueError("Key must be in format 'eye_variable'")
        return key

    @property
    def mask(self) -> Mapping:
        """Mask of each key under the default mask policy (read-only
        mapping key -> int array, 1: masked; see `get_policy_mask`).

        The arrays are read-only.
        """
        return _PolicyMask(self)

    # --- layers -----------------------------------------------------------

    def layer_bits(self, names, create: bool = False) -> int:
        """
        Bit mask of the layers `names` (0 for layers that are not registered).

        Parameters
        ----------
        names : str or list of str
            layer name(s)
        create : bool, optional
            register layers that are not registered yet (next free bit)

        Returns
        -------
        int
            the bits of the layers OR-ed together

        Raises
        ------
        ValueError
            if a new layer is needed but all `MAX_LAYERS` bits are used
        """
        if isinstance(names, str):
            names = [names]
        bits = 0
        for name in names:
            if name not in self.layer_names:
                if not create:
                    continue
                if len(self.layer_names) >= MAX_LAYERS:
                    raise ValueError(f"cannot add layer {name!r}: all {MAX_LAYERS} layers "
                                     f"are used ({', '.join(self.layer_names)})")
                self.layer_names.append(name)
            bits |= 1 << self.layer_names.index(name)
        return bits

    def get_layer(self, key: str, name: str) -> NDArray:
        """Boolean array that is True where layer `name` is set for `key`."""
        return (self.layers[key] & self.layer_bits(name)) != 0

    def set_layer(self, key: str, name: str, mask: NDArray, replace: bool = False) -> None:
        """
        Set layer `name` of `key` where `mask` is True.

        Parameters
        ----------
        key : str
            key of the data (e.g., "left_pupil")
        name : str
            layer name (registered if new)
        mask : array-like of bool
            one value per sample
        replace : bool, optional
            clear the layer first (True) or add to it (False, default)
        """
        if key not in self.data:
            raise KeyError(f"No data for key {key}")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"Mask must have shape {self.shape}")
        bit = LAYER_DTYPE(self.layer_bits(name, create=True))
        if replace:
            self.layers[key] &= ~bit
        self.layers[key][mask] |= bit

    def clear_layer(self, key: str, name: str, where: Optional[NDArray] = None) -> None:
        """Clear layer `name` of `key` everywhere or at the samples `where`
        (an index, e.g. a boolean array or ``np.s_[10:20]``)."""
        bit = LAYER_DTYPE(self.layer_bits(name))
        if bit:
            if where is None:
                self.layers[key] &= ~bit
            else:
                self.layers[key][where] &= ~bit

    def get_rec_layer(self, name: str) -> NDArray:
        """Boolean array that is True where the recording-wide layer `name` is set."""
        if self.rec_layers is None:
            return np.zeros(self.length, dtype=bool)
        return (self.rec_layers & self.layer_bits(name)) != 0

    def set_rec_layer(self, name: str, mask: NDArray, replace: bool = False) -> None:
        """
        Set the recording-wide layer `name` (all keys) where `mask` is True.

        Parameters
        ----------
        name : str
            layer name (registered if new)
        mask : array-like of bool
            one value per sample
        replace : bool, optional
            clear the layer first (True) or add to it (False, default)
        """
        if self.rec_layers is None:
            raise KeyError("No data")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"Mask must have shape {self.shape}")
        bit = LAYER_DTYPE(self.layer_bits(name, create=True))
        if replace:
            self.rec_layers &= ~bit
        self.rec_layers[mask] |= bit

    def clear_rec_layer(self, name: str) -> None:
        """Clear the recording-wide layer `name`."""
        bit = LAYER_DTYPE(self.layer_bits(name))
        if bit and self.rec_layers is not None:
            self.rec_layers &= ~bit

    def get_policy_mask(self, key: str, policy="default") -> NDArray:
        """
        Boolean mask of `key` under `policy` (True: unusable sample).

        Parameters
        ----------
        key : str
            key of the data (e.g., "left_pupil")
        policy : str, MaskPolicy or list of str
            a predefined policy (see `POLICIES`), a :class:`MaskPolicy`, or
            a list of layers (each of them masks a sample; NaN as well)

        Raises
        ------
        ValueError
            if the policy names a layer that is neither a standard layer
            nor registered
        """
        policy = _as_policy(policy)
        unknown = [name for name in policy.layers + policy.unless
                   if name not in STANDARD_LAYERS and name not in self.layer_names
                   and not name.startswith("_")]  # internal layers
        if unknown:
            raise ValueError(f"unknown layer(s) {', '.join(unknown)}; registered layers: "
                             f"{', '.join(self.layer_names) or 'none'}")
        layers = self.layers[key]
        masked = (layers & self.layer_bits(policy.layers)) != 0
        unless = self.layer_bits(policy.unless)
        if unless:
            masked &= (layers & unless) == 0
        # recording-wide layers (e.g., excluded): `unless` does not apply
        masked |= (self.rec_layers & self.layer_bits(policy.layers)) != 0
        if policy.nan:
            masked |= np.isnan(self.data[key])
        return masked

    def _set_state(self, st: dict) -> None:
        """Set data and layers from a state; length and shape follow from them."""
        super()._set_state(st)
        self.layer_names = [str(name) for name in self.layer_names]
        first = next(iter(self.data.values()), None)
        self.length = 0 if first is None else first.shape[0]
        self.shape = None if first is None else first.shape
        if self.rec_layers is not None:
            self.rec_layers = np.asarray(self.rec_layers, dtype=LAYER_DTYPE)
        elif first is not None:
            self._init_rec_layers()
        if "_nan" in self.layer_names:
            self._drop_nan_layer()

    def _drop_nan_layer(self) -> None:
        """Remove the layer ``"_nan"`` of states written by development
        versions of 2.2 (format 3 before the mask policy was used for all
        processing): it marked samples that were NaN when they were set.
        Where such a sample holds a value now (filled by a filter) and is
        not interpolated, it stays masked (``"user"`` layer)."""
        for key, layers in self.layers.items():
            filled = (self.get_layer(key, "_nan") & ~self.get_layer(key, "interpolated")
                      & ~np.isnan(self.data[key]))
            if filled.any():
                self.set_layer(key, "user", filled)
        i = self.layer_names.index("_nan")
        low, high = (1 << i) - 1, ~((1 << (i + 1)) - 1) & 0xFFFF

        def drop(arr):
            return ((arr & low) | ((arr & high) >> 1)).astype(LAYER_DTYPE)

        self.layers = {key: drop(arr) for key, arr in self.layers.items()}
        if self.rec_layers is not None:
            self.rec_layers = drop(self.rec_layers)
        del self.layer_names[i]

    def copy(self) -> 'EyeDataDict':
        """Create a deep copy of the dictionary (data and masks)."""
        new_dict = EyeDataDict()
        new_dict.layer_names = list(self.layer_names)
        for key, value in self.data.items():
            new_dict._set_channel(key, value.copy(), self.layers[key])
        if self.rec_layers is not None:
            new_dict.rec_layers = self.rec_layers.copy()
        return new_dict

    def set_mask(self, key: str, mask: NDArray, layer: str = "user") -> None:
        """Set mask for a specific key.

        Afterwards, ``self.mask[key]`` (the mask under the default policy)
        equals `mask`: samples that are not masked in `mask` are removed
        from the layers of the key that mask them, samples that were not
        masked yet are added to `layer` (default ``"user"``; where they are
        interpolated, the ``"interpolated"`` layer is cleared, as it would
        make them usable). NaN values and the recording-wide layers (e.g.,
        excluded periods) are not changed, so these samples stay masked.

        Parameters
        ----------
        key : str
            Key of the data (e.g., "left_pupil")
        mask : array-like
            Mask array (1 for masked, 0 for valid values) with the same shape
            as the data; converted to int
        layer : str, optional
            layer for the newly masked samples (default ``"user"``)

        Raises
        ------
        KeyError
            if there is no data for `key`
        ValueError
            if `mask` does not have the same shape as the data
        """
        if key not in self.data:
            raise KeyError(f"No data for key {key}")
        mask = np.asarray(mask)
        if mask.shape != self.shape:
            raise ValueError(f"Mask must have shape {self.shape}")
        mask = mask.astype(bool)
        default = POLICIES["default"]
        bits = LAYER_DTYPE(self.layer_bits(default.layers))
        self.layers[key][~mask] &= ~bits
        new = mask & ~self.get_policy_mask(key)
        if new.any():
            for name in default.unless:
                self.clear_layer(key, name, new)
            self.set_layer(key, layer, new)

    def add_mask(self, key: str, where) -> None:
        """Mask the samples `where` of `key` (``"user"`` layer; existing masks are kept).

        Parameters
        ----------
        key : str
            Key of the data (e.g., "left_pupil")
        where : array-like of bool, slice, int or array of int
            the samples to mask: a boolean array (one value per sample,
            True: mask the sample) or any index, e.g. ``np.s_[10:20]``

        Raises
        ------
        KeyError
            if there is no data for `key`
        """
        if key not in self.data:
            raise KeyError(f"No data for key {key}")
        mask = np.zeros(self.shape, dtype=bool)
        mask[where] = True
        if mask.any():
            self.set_layer(key, "user", mask)

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
        for arr in self.layers.values():
            total_size += arr.nbytes
        if self.rec_layers is not None:
            total_size += self.rec_layers.nbytes
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
            return {key: int(np.sum(mask)) for key, mask in self.mask.items()}
        else:
            return max(int(np.sum(mask)) for mask in self.mask.values())

