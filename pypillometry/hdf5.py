"""
Store the state of an object (see :mod:`pypillometry.state`) in an HDF5 file.

The state dict is mapped to the file as follows:

- dicts -> groups (in the order of their keys)
- numpy arrays -> datasets (compressed with gzip when larger than
  `_COMPRESS_MIN_SIZE` elements); text arrays are stored as UTF-8 strings,
  their numpy dtype in the attribute ``__dtype__``
- lists and tuples -> datasets if all elements are numbers, bools or all
  strings, otherwise groups with the elements as ``"0"``, ``"1"``, ...;
  both are marked with the attribute ``__type__`` (``"list"`` or ``"tuple"``)
- numbers, bools and strings -> attributes of the group (texts longer
  than `_MAX_ATTR_TEXT` characters: scalar datasets with ``__type__``
  ``"str"``)
- None -> an empty attribute
- datetime/date -> ISO text in an attribute (type noted in ``__types__``)
- anything else -> its ``repr()`` as text (with a warning; type ``"repr"``
  in ``__types__``)

The root group has the attributes ``format_version`` and ``class`` (part of
the state) and ``pypillometry_version`` (the version that wrote the file).
Numbers are read back as Python numbers (numpy scalars become ``float``,
``int`` or ``bool``).
"""

import datetime
import json

import h5py
import numpy as np
from loguru import logger

__all__ = []

# arrays with fewer elements are stored without compression
_COMPRESS_MIN_SIZE = 1000

# attributes with metadata (not part of the state)
_TYPE = "__type__"      # "list" or "tuple" for groups/datasets holding one
_DTYPE = "__dtype__"    # numpy dtype of a text array
_TYPES = "__types__"    # JSON dict: attribute name -> "datetime"/"date"/"repr"
_ORDER = "__order__"    # JSON list of the keys (if attributes and members mix)
_META = {_TYPE, _DTYPE, _TYPES, _ORDER, "pypillometry_version"}

_STR_DTYPE = h5py.string_dtype("utf-8")

# longer texts are stored as datasets (attributes are limited to 64 kB;
# UTF-8 needs up to 4 bytes per character)
_MAX_ATTR_TEXT = 8000


def write_state(st: dict, fname: str, compress: bool = True):
    """
    Write a state dict to the HDF5 file `fname` (overwritten if it exists).

    Parameters
    ----------
    st : dict
        state (see :func:`pypillometry.state.to_state`)
    fname : str
        filename
    compress : bool, optional
        if True (default), compress larger arrays with gzip
    """
    # the file format of HDF5 1.10 stores small groups more compactly
    with h5py.File(fname, "w", track_order=True, libver="v110") as f:
        f.attrs["pypillometry_version"] = _package_version()
        _write_dict(f, st, compress, "")


def read_state(fname: str) -> dict:
    """
    Read a state dict written by :func:`write_state`.

    Parameters
    ----------
    fname : str
        filename

    Returns
    -------
    dict
        the state (in the version it was written in)
    """
    with h5py.File(fname, "r") as f:
        return _read_group(f)


def _package_version():
    """Installed version of pypillometry (or "unknown")."""
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("pypillometry")
    except PackageNotFoundError:
        return "unknown"


def is_hdf5(fname: str) -> bool:
    """True if `fname` starts with the HDF5 signature."""
    with open(fname, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def _is_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) \
        and not isinstance(v, (bool, np.bool_))


def _is_bool(v):
    return isinstance(v, (bool, np.bool_))


def _valid_keys(d):
    return all(isinstance(k, str) and k and "/" not in k and k != "."
               and k not in _META for k in d)


def _write_dict(group, d, compress, path):
    types = {}
    for key, value in d.items():
        kind = _write_value(group, key, value, compress, f"{path}/{key}")
        if kind is not None:
            types[key] = kind
    if types:
        group.attrs[_TYPES] = json.dumps(types)
    # attributes and members are ordered separately; keep the key order
    attrs = [k for k in group.attrs if k not in _META]
    if attrs and len(group) and list(d) != attrs + list(group):
        group.attrs[_ORDER] = json.dumps(list(d))


def _write_value(group, key, value, compress, path):
    """Write `value` as `key` into `group`; return the type for ``__types__`` (or None)."""
    if value is None:
        group.attrs[key] = h5py.Empty("f")
    elif isinstance(value, str):
        _write_text(group, key, value)
    elif _is_bool(value) or _is_number(value) or isinstance(value, complex):
        group.attrs[key] = value
    elif isinstance(value, datetime.datetime):
        group.attrs[key] = value.isoformat()
        return "datetime"
    elif isinstance(value, datetime.date):
        group.attrs[key] = value.isoformat()
        return "date"
    elif isinstance(value, dict) and _valid_keys(value):
        _write_dict(group.create_group(key, track_order=True), value, compress, path)
    elif isinstance(value, (list, tuple)):
        _write_sequence(group, key, value, compress, path)
    elif isinstance(value, np.ndarray) and not isinstance(value, np.ma.MaskedArray) \
            and _write_array(group, key, value, compress):
        pass
    else:
        logger.warning(f"Cannot store {path} ({type(value).__name__}) in the "
                       f"HDF5 file; storing its text representation instead")
        _write_text(group, key, repr(value))
        return "repr"
    return None


def _write_text(group, key, text):
    """Text as attribute, or as scalar dataset if too long for an attribute."""
    if len(text) <= _MAX_ATTR_TEXT:
        group.attrs[key] = text
    else:
        ds = group.create_dataset(key, data=text, dtype=_STR_DTYPE)
        ds.attrs[_TYPE] = "str"


def _write_array(group, key, arr, compress):
    """Write a numpy array as dataset; False if its dtype is not supported."""
    attrs = {}
    if arr.dtype.kind == "U" or (arr.dtype.kind == "O" and
                                 all(isinstance(x, str) for x in arr.flat)):
        attrs[_DTYPE] = arr.dtype.str
        arr = arr.astype(object)
        dtype = _STR_DTYPE
    elif arr.dtype.kind in "biufcSV" and not arr.dtype.hasobject:
        dtype = arr.dtype
    else:
        return False
    kwargs = {}
    if compress and arr.size >= _COMPRESS_MIN_SIZE:
        kwargs = dict(compression="gzip", compression_opts=4, shuffle=True)
    ds = group.create_dataset(key, data=arr, dtype=dtype, **kwargs)
    ds.attrs.update(attrs)
    return True


def _write_sequence(group, key, seq, compress, path):
    """Write a list/tuple as dataset (numbers, bools or strings) or group."""
    kind = "tuple" if isinstance(seq, tuple) else "list"
    simple = len(seq) > 0 and (all(isinstance(x, str) for x in seq)
                               or all(_is_bool(x) for x in seq)
                               or all(isinstance(x, (int, np.integer)) and not _is_bool(x)
                                      for x in seq)
                               or all(isinstance(x, (float, np.floating)) for x in seq))
    if simple:
        arr = np.array(seq, dtype=object if isinstance(seq[0], str) else None)
        ds = group.create_dataset(key, data=arr,
                                  dtype=_STR_DTYPE if isinstance(seq[0], str) else arr.dtype)
        ds.attrs[_TYPE] = kind
    else:
        sub = group.create_group(key, track_order=True)
        sub.attrs[_TYPE] = kind
        _write_dict(sub, {str(i): v for i, v in enumerate(seq)}, compress, path)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def _read_group(group):
    """Dict (or list/tuple) with the content of `group`."""
    types = json.loads(group.attrs.get(_TYPES, "{}"))
    # attributes first, then members; __order__ restores the key order
    items = {}
    for key, value in group.attrs.items():
        if key not in _META:
            items[key] = _read_attr(value, types.get(key))
    for key, node in group.items():
        items[key] = _read_group(node) if isinstance(node, h5py.Group) else _read_dataset(node)
    order = group.attrs.get(_ORDER)
    d = items if order is None else {k: items[k] for k in json.loads(order)}
    kind = group.attrs.get(_TYPE)
    if kind is not None:
        seq = [d[str(i)] for i in range(len(d))]
        return tuple(seq) if kind == "tuple" else seq
    return d


def _read_attr(value, kind):
    if isinstance(value, h5py.Empty):
        return None
    if kind == "datetime":
        return datetime.datetime.fromisoformat(value)
    if kind == "date":
        return datetime.date.fromisoformat(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_dataset(ds):
    kind = ds.attrs.get(_TYPE)
    dtype = ds.attrs.get(_DTYPE)
    if kind == "str":  # long text
        return ds.asstr()[()]
    if dtype is not None:  # text array
        return np.asarray(ds.asstr()[()], dtype=object).astype(dtype)
    if kind is None:  # numeric (or other) array
        arr = ds[()]
        return np.asarray(arr) if ds.shape == () else arr
    if h5py.check_string_dtype(ds.dtype) is not None:  # list of strings
        seq = ds.asstr()[()].tolist()
    else:
        seq = ds[()].tolist()
    return tuple(seq) if kind == "tuple" else seq
