"""
Versioned state of the objects that can be saved to a file.

The eyedata objects (:class:`~pypillometry.EyeData`,
:class:`~pypillometry.FakeEyeData`) and the
segmented data (:class:`~pypillometry.SegmentedEyeData`, ...) define their
content as a *state*: a dict that contains only numbers, strings, None,
lists, tuples, dicts and numpy arrays (plus whatever the user stored in
`info` and `params`). Its keys ``format_version`` and ``class`` say which
version of the layout the state has and which class it belongs to.

Each class declares the attributes that make up its state in
`_STATE_FIELDS` (see :class:`Stateful`); objects of other classes inside
it (e.g., :class:`~pypillometry.Intervals`) are stored as nested states.
A test checks that every attribute of an object is part of its state.

The file formats are thin layers around the state:

- pickle: ``__getstate__`` returns the state and ``__setstate__`` restores
  the object from it
- HDF5: the state is written to the file (see :mod:`pypillometry.hdf5`)

When the layout of a state changes, :data:`FORMAT_VERSION` is increased and
a function that upgrades a state by one version is added to `_MIGRATIONS`.
Old states are upgraded with :func:`migrate` before the object is restored,
regardless of the file format.

Version history
---------------
0
    pickles written before the state was versioned (pypillometry <= 2.0):
    the raw attributes of the object (its ``__dict__``)
1
    the state as declared in `_STATE_FIELDS` of the classes
2
    Intervals without `sampling_rate` and `time_offset` (conversions
    between sample indices and time use the data object); sample indices
    are ``units="samples"`` (instead of None); eyedata objects have
    excluded periods (`excluded`, Intervals in ms); metadata of the
    events: eyedata objects have `event_meta` (dict of arrays, one value
    per event), Intervals have `event_meta` (None or dict of arrays, one
    value per interval)
3
    eyedata objects store their events as one Events state (`events`:
    `onsets` in `units`, `labels`, `data_time_range`, `meta`); all eyedata
    objects have `experimental_setup` and `calibration`; masks are
    layers: EyeDataDicts have `layers` (one uint16 bit-field per key),
    `layer_names` (layer i is bit i) and `rec_layers` (recording-wide
    layers, e.g. excluded periods) instead of `mask`; eyedata objects have
    no `blinks`, `interpolated_blinks` and `excluded` (they are layers);
    Intervals have no `event_indices`, `data_time_range` and
    `interval_window`; segments store their Epochs (`onsets` and `window`
    in ms, `labels`, `meta`, `label`) or, if not event-locked, Intervals,
    with the class name in ``intervals["class"]``; the response parameters
    (``params["response"]``) are keyed by channel (``"left_pupil"``)
"""

import contextlib
import importlib
import pickle
from collections import namedtuple

import numpy as np
from loguru import logger

__all__ = []

#: version of the state written by this version of pypillometry
FORMAT_VERSION = 3

# classes that moved: (old module, old name) -> (new module, new name),
# used when reading pickle files (see _Unpickler) and states (see _find_class)
_RENAMED_CLASSES = {
    # one class for all eyedata (pypillometry 2.2)
    ("pypillometry.eyedata.pupildata", "PupilData"): ("pypillometry.eyedata.eyedata", "EyeData"),
    ("pypillometry.eyedata.gazedata", "GazeData"): ("pypillometry.eyedata.eyedata", "EyeData"),
}

# list that collects (class name, format version) of the states passed to
# migrate(), or None (see _recording_versions)
_versions_seen = None


def to_state(obj) -> dict:
    """
    Return the state of `obj` (a dict with ``format_version`` and ``class``).

    Parameters
    ----------
    obj : object
        object with a ``_to_state()`` method (eyedata or segmented data)

    Returns
    -------
    dict
        the state of the object
    """
    if not hasattr(obj, "_to_state"):
        raise TypeError(f"objects of type {type(obj).__name__} have no state")
    return {"format_version": FORMAT_VERSION, "class": type(obj).__name__,
            **obj._to_state()}


def from_state(state: dict, cls=None):
    """
    Create an object from its state (upgraded to the current version first).

    Parameters
    ----------
    state : dict
        state as returned by :func:`to_state` (of this or an older version)
    cls : type, optional
        class of the object; if None, the class is looked up from the
        ``class`` entry of the state

    Returns
    -------
    object
        the restored object
    """
    if cls is None:
        cls = _find_class(state["class"])
    obj = cls.__new__(cls)
    obj.__setstate__(state)
    return obj


def migrate(state: dict, class_name: str) -> dict:
    """
    Upgrade `state` to the current :data:`FORMAT_VERSION`.

    A dict without ``format_version`` is treated as version 0 (the raw
    attributes of an object pickled before the state was versioned).

    Parameters
    ----------
    state : dict
        state of any version up to the current one
    class_name : str
        name of the class of the object (needed for version 0, whose states
        do not contain it)

    Returns
    -------
    dict
        the state in the current version (may be `state` itself)
    """
    version = state.get("format_version", 0)
    if _versions_seen is not None:
        _versions_seen.append((class_name, version))
    if version > FORMAT_VERSION:
        raise ValueError(
            f"the data was saved with a newer version of pypillometry (format "
            f"version {version}, this version reads up to {FORMAT_VERSION}); "
            f"please update pypillometry")
    if version < FORMAT_VERSION:
        logger.debug(f"Upgrading {class_name} state from format version "
                     f"{version} to {FORMAT_VERSION}")
    while version < FORMAT_VERSION:
        state = _MIGRATIONS[version](state, class_name)
        version += 1
        state["format_version"] = version
    return state


@contextlib.contextmanager
def _recording_versions():
    """
    Context manager that collects the format versions of the objects loaded
    in its block.

    Yields a list of ``(class name, format version)``, one entry for each
    state passed to :func:`migrate` (each object, including nested ones such
    as `original`), before it is upgraded.
    """
    global _versions_seen
    outer, _versions_seen = _versions_seen, []
    try:
        yield _versions_seen
    finally:
        _versions_seen = outer


def _find_class(name: str):
    """The class (eyedata or segmented data) called `name` (or renamed from `name`)."""
    renamed = {old: new for (_, old), (_, new) in _RENAMED_CLASSES.items()}
    name = renamed.get(name, name)
    from .eyedata.generic import GenericEyeData
    from .segmented import SegmentedEyeData
    # importing the package registers all its classes as subclasses
    from . import eyedata, segmented  # noqa: F401
    todo = [GenericEyeData, SegmentedEyeData]
    while todo:
        cls = todo.pop()
        if cls.__name__ == name:
            return cls
        todo.extend(cls.__subclasses__())
    raise ValueError(f"unknown class {name!r}")


def _function_name(fct):
    """``"module:qualname"`` of a function (to import it again), or None."""
    if fct is None:
        return None
    return f"{fct.__module__}:{fct.__qualname__}"


def _import_function(name):
    """The function called `name` (``"module:qualname"``), or None."""
    if name is None:
        return None
    module, _, qualname = name.partition(":")
    try:
        obj = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    except (ImportError, AttributeError):
        logger.warning(f"Could not import function {name}; set to None")
        return None


class _Unpickler(pickle.Unpickler):
    """Unpickler that maps classes that moved to their new location."""

    def find_class(self, module, name):
        module, name = _RENAMED_CLASSES.get((module, name), (module, name))
        return super().find_class(module, name)


# --------------------------------------------------------------------------
# Declaring the state of a class
# --------------------------------------------------------------------------

Field = namedtuple("Field", "key kind attr")


def _field(spec):
    """Field from its spec: ``"name"``, ``(key, kind)`` or ``(key, kind, attribute)``."""
    if isinstance(spec, str):
        spec = (spec,)
    key, kind, attr = (tuple(spec) + (None, None))[:3]
    return Field(key, kind, attr or key)


class Stateful:
    """
    Mixin for classes whose state is declared in `_STATE_FIELDS`.

    Each class lists the fields it adds to the state of its base classes.
    A field is the name of an attribute (stored as it is) or a tuple
    ``(key, kind)`` or ``(key, kind, attribute)``; `kind` says how the value
    is converted (see `_KINDS`; ``"dict:<kind>"`` converts the values of a
    dict). None is stored as None for every kind. Attributes that are not
    part of the state (e.g., derived ones) are listed in `_STATE_EXCLUDE`.
    """
    _STATE_FIELDS = ()
    _STATE_EXCLUDE = ()

    @classmethod
    def _state_fields(cls):
        """All fields of the class (those of the base classes first)."""
        fields = {}
        for c in reversed(cls.__mro__):
            for spec in c.__dict__.get("_STATE_FIELDS", ()):
                f = _field(spec)
                fields[f.key] = f
        return list(fields.values())

    @classmethod
    def _state_exclude(cls):
        """Attributes that are not part of the state (of all base classes)."""
        return {a for c in cls.__mro__ for a in c.__dict__.get("_STATE_EXCLUDE", ())}

    def _to_state(self) -> dict:
        """State of the object (without ``format_version`` and ``class``)."""
        return {f.key: _convert(f.kind, getattr(self, f.attr), 0)
                for f in self._state_fields()}

    def _set_state(self, st: dict):
        """Set the attributes from a state of the current version."""
        for f in self._state_fields():
            setattr(self, f.attr, _convert(f.kind, st[f.key], 1))

    @classmethod
    def _from_state(cls, st: dict):
        """Create an object from a state of the current version."""
        obj = cls.__new__(cls)
        obj._set_state(st)
        return obj


class VersionedState(Stateful):
    """
    Mixin for classes that are saved to files: pickles store the versioned
    state (see :func:`to_state`), older versions are upgraded when loaded.
    """

    def __getstate__(self):
        return to_state(self)

    def __setstate__(self, st):
        self._set_state(migrate(st, type(self).__name__))


def _nested_class(name):
    """Classes stored as nested states (imported here to avoid import cycles)."""
    from .intervals import Intervals
    from .eyedata.eyedatadict import EyeDataDict
    from .eyedata.experimental_setup import ExperimentalSetup
    from .eyedata.spatial_calibration import SpatialCalibration
    from .events import Events
    from .epochs import Epochs
    return {"Intervals": Intervals, "EyeDataDict": EyeDataDict, "Events": Events,
            "Epochs": Epochs,
            "ExperimentalSetup": ExperimentalSetup,
            "SpatialCalibration": SpatialCalibration}[name]


def _pairs_to_state(pairs):
    return np.array(pairs).reshape(len(pairs), 2)


def _pairs_from_state(arr):
    return [tuple(row) for row in np.asarray(arr).tolist()]


def _tagged_to_state(obj):
    return {"class": type(obj).__name__, **obj._to_state()}


def _tagged_from_state(st):
    st = dict(st)
    return _nested_class(st.pop("class"))._from_state(st)


def _masked_to_state(arr):
    return {"values": np.ma.getdata(arr), "mask": np.ma.getmaskarray(arr)}


def _masked_from_state(st):
    return np.ma.masked_array(st["values"], mask=st["mask"])


# kind -> (value to state, state to value)
_KINDS = {
    # whole object with its own format_version and class (e.g., `original`)
    "versioned": (to_state, from_state),
    # list of (start, end) tuples <-> (n, 2) array
    "pairs": (_pairs_to_state, _pairs_from_state),
    # masked array <-> {"values": ..., "mask": ...}
    "masked": (_masked_to_state, _masked_from_state),
    # function <-> "module:qualname"
    "function": (_function_name, _import_function),
    # object of one of the classes in _nested_class() <-> its state with
    # its class name in "class" (for attributes that can hold several classes)
    "tagged": (_tagged_to_state, _tagged_from_state),
}


def _convert(kind, value, direction):
    """Convert `value` to its state (direction 0) or back (direction 1)."""
    if kind is None or value is None:
        return value
    if kind.startswith("dict:"):
        return {k: _convert(kind[5:], v, direction) for k, v in value.items()}
    if kind in _KINDS:
        return _KINDS[kind][direction](value)
    # nested object of one of the classes in _nested_class()
    return value._to_state() if direction == 0 else _nested_class(kind)._from_state(value)


# --------------------------------------------------------------------------
# Migrations: _MIGRATIONS[v] upgrades a state from version v to v + 1.
# These functions are frozen: they describe the layout at that version and
# must not change when the classes change later.
# --------------------------------------------------------------------------

_SEGMENTED_CLASSES = ("SegmentedEyeData", "GroupLevelSegmentedData")


def _v0_dict(obj):
    """Raw attributes of an object from a version-0 pickle (or None)."""
    return None if obj is None else vars(obj)


def _v0_eyedatadict(raw):
    """Version-1 state of an EyeDataDict (raw attributes `raw`)."""
    return {"values": dict(raw["data"]), "mask": dict(raw["mask"])}


def _v0_intervals(raw):
    """Version-1 state of an Intervals object (raw attributes `raw`)."""
    if raw is None:
        return None
    ivs = raw["intervals"]
    return {
        "intervals": np.array(ivs).reshape(len(ivs), 2),
        "units": raw.get("units"),
        "label": raw.get("label"),
        "event_labels": raw.get("event_labels"),
        "event_indices": raw.get("event_indices"),
        "event_onsets": raw.get("event_onsets"),
        "data_time_range": raw.get("data_time_range"),
        "sampling_rate": raw.get("sampling_rate"),
        "time_offset": raw.get("time_offset", 0.0),
        "interval_window": raw.get("interval_window"),
    }


def _v0_setup(raw):
    """Version-1 state of an ExperimentalSetup (raw attributes `raw`)."""
    if raw is None:
        return None
    keys = ("screen_resolution", "physical_screen_size", "screen_offset", "d",
            "alpha_tilt", "beta_tilt", "camera_offset", "camera_spherical",
            "ipd")
    return {k: raw.get("_" + k) for k in keys}


def _v0_calibration(raw):
    """Version-1 state of a SpatialCalibration (raw attributes `raw`)."""
    keys = ("eye", "model", "points", "measured", "errors",
            "screen_resolution")
    return {k: raw.get(k) for k in keys}


def _migrate_0_to_1(raw, class_name):
    """Version 0 (raw attributes) -> version 1 (state)."""
    if class_name in _SEGMENTED_CLASSES:
        data = raw["data"]
        state = {
            "class": class_name,
            "name": raw["name"],
            "variable": raw["variable"],
            "tx": raw["tx"],
            "data": {"values": np.ma.getdata(data), "mask": np.ma.getmaskarray(data)},
            "intervals": _v0_intervals(_v0_dict(raw.get("intervals"))),
            "sampling_rate": raw.get("sampling_rate"),
        }
        if class_name == "GroupLevelSegmentedData":
            state["mask_percent"] = raw["mask_percent"]
            state["n_subjects"] = raw["n_subjects"]
        return state

    # eyedata classes (and subclasses defined by users)
    original = raw.get("original")
    if original is not None:
        # already restored by its own __setstate__, so its state has the
        # current version (each eyedata state is migrated on its own)
        original = to_state(original)
    blinks = raw.get("_blinks") or {}
    iblinks = raw.get("_interpolated_blinks") or {}
    state = {
        "class": class_name,
        "name": raw["name"],
        "fs": raw["fs"],
        "tx": raw["tx"],
        "data": _v0_eyedatadict(vars(raw["data"])),
        "event_onsets": raw["event_onsets"],
        "event_labels": raw["event_labels"],
        "blinks": {k: _v0_intervals(_v0_dict(v)) for k, v in blinks.items()},
        "interpolated_blinks": {k: _v0_intervals(_v0_dict(v))
                                for k, v in iblinks.items()},
        "params": raw.get("params", {}),
        # entries before pypillometry 2.0 also had the arguments as objects
        "history": [{k: v for k, v in ev.items() if k not in ("args", "kwargs")}
                    for ev in raw.get("history", [])],
        "inplace": raw.get("inplace"),
        "info": raw.get("info", {}),
        "original": original,
    }
    # gaze data (PupilData made from EyeData kept these attributes; dropped)
    if class_name != "PupilData" and (
            "experimental_setup" in raw or "calibration" in raw):
        cal = raw.get("calibration")
        state["experimental_setup"] = _v0_setup(
            _v0_dict(raw.get("experimental_setup")))
        state["calibration"] = None if cal is None else {
            eye: _v0_calibration(vars(c)) for eye, c in cal.items()}
    if "sim_data" in raw:  # FakeEyeData
        state["sim_fct"] = _function_name(raw.get("sim_fct"))
        state["sim_fct_name"] = raw.get("sim_fct_name")
        state["sim_params"] = raw.get("sim_params", {})
        state["sim_data"] = _v0_eyedatadict(vars(raw["sim_data"]))
    return state


def _v1_intervals(st):
    """Version-2 state of an Intervals object (version-1 state `st`)."""
    if st is None:
        return None
    st = {k: v for k, v in st.items() if k not in ("sampling_rate", "time_offset")}
    if st.get("units") is None:
        st["units"] = "samples"
    st["event_meta"] = None
    return st


def _migrate_1_to_2(state, class_name):
    """Version 1 -> 2: Intervals lose `sampling_rate` and `time_offset`;
    sample indices are "samples" instead of None; no excluded periods;
    no metadata of events and Intervals."""
    state = dict(state)
    if class_name in _SEGMENTED_CLASSES:
        state["intervals"] = _v1_intervals(state.get("intervals"))
        return state
    for key in ("blinks", "interpolated_blinks"):
        if state.get(key) is not None:
            state[key] = {k: _v1_intervals(v) for k, v in state[key].items()}
    # no excluded periods
    state["excluded"] = {"intervals": np.zeros((0, 2)), "units": "ms", "label": "excluded",
                         "event_labels": None, "event_indices": None, "event_onsets": None,
                         "data_time_range": None, "interval_window": None,
                         "event_meta": None}
    state["event_meta"] = {}
    return state


# layers that made up the mask of version 2 (the "compat" mask of version 3)
_V2_MASK_LAYERS = ("user", "interpolated", "blink", "excluded",
                   "offscreen", "divergence")
_V2_MS_PER_UNIT = {"ms": 1., "sec": 1000., "min": 60000., "h": 3600000.}


class _V3Layers:
    """Layers of a version-3 EyeDataDict state (`st`, changed in place)."""

    def __init__(self, st):
        self.st = st

    def bit(self, name):
        """Bit of layer `name` (registered if new)."""
        names = self.st["layer_names"]
        if name not in names:
            if len(names) == 16:
                raise ValueError("more than 16 mask layers")
            names.append(name)
        return np.uint16(1 << names.index(name))

    def bits(self, names):
        """Bits of the registered layers among `names`."""
        return np.uint16(sum(1 << i for i, n in enumerate(self.st["layer_names"])
                             if n in names))

    def get(self, key, name):
        """Layer `name` of `key` (None: the recording-wide layer) as bool array."""
        arr = self.st["rec_layers"] if key is None else self.st["layers"][key]
        return (arr & self.bits([name])) != 0

    def set(self, key, name, where):
        """Set layer `name` of `key` (None: recording-wide) where `where` is True."""
        arr = self.st["rec_layers"] if key is None else self.st["layers"][key]
        arr[where] |= self.bit(name)

    def clear(self, key, name, where):
        self.st["layers"][key][where] &= ~self.bits([name])

    def mask(self, key):
        """The mask of `key` as version 2 had it (from the layers and NaN)."""
        bits = self.bits(_V2_MASK_LAYERS)
        return (((self.st["layers"][key] & bits) != 0) | ((self.st["rec_layers"] & bits) != 0)
                | np.isnan(self.st["values"][key]))


def _v2_eyedatadict(st):
    """Version-3 state of an EyeDataDict (version-2 state `st`), with its _V3Layers.

    The mask of version 2 (``mask``) becomes layers: masked samples that are
    not NaN the ``"user"`` layer (NaN values are masked by their value).
    States written by development versions between 2.1 and 3 may already
    have (some) layers.
    """
    st = dict(st)
    values = st["values"]
    if st.get("layers") is None:
        st["layers"] = {k: np.zeros(np.shape(v), dtype=np.uint16) for k, v in values.items()}
    else:
        st["layers"] = {k: np.array(v, dtype=np.uint16) for k, v in st["layers"].items()}
    st["layer_names"] = [str(name) for name in st.get("layer_names") or []]
    if st.get("rec_layers") is not None:
        st["rec_layers"] = np.array(st["rec_layers"], dtype=np.uint16)
    elif values:
        st["rec_layers"] = np.zeros(np.shape(next(iter(values.values()))), dtype=np.uint16)
    else:
        st["rec_layers"] = None
    layers = _V3Layers(st)
    for key, m in (st.pop("mask", None) or {}).items():
        m = np.asarray(m).astype(bool) & ~layers.mask(key)
        if m.any():
            layers.set(key, "user", m)
    return st, layers


def _v2_intervals_mask(st, tx):
    """Samples (bool array) covered by an Intervals state `st` of version 2.

    Times are converted to the first sample at or after them (as Intervals
    did in version 2); the intervals are clipped to the recording.
    """
    n = len(tx)
    out = np.zeros(n, dtype=bool)
    ivs = np.asarray(st["intervals"], dtype=float).reshape(-1, 2)
    if st["units"] != "samples" and n > 0:
        tol = 1e-6 * (np.median(np.diff(tx)) if n > 1 else 1.)
        ivs = np.searchsorted(tx, ivs * _V2_MS_PER_UNIT[st["units"]] - tol, side="left")
    for start, end in ivs:
        out[min(max(int(start), 0), n):min(max(int(end), 0), n)] = True
    return out


def _v2_spans(st, fs):
    """Version-3 state (Epochs or Intervals, with "class") of the version-2
    Intervals state `st` of segments (sampling rate `fs`).

    Event-locked intervals (with `interval_window` and `event_onsets`) become
    Epochs (in ms); onsets in sample indices are converted with `fs` (from the
    first sample of the recording, whose time is unknown).
    """
    if st is None:
        return None
    locked = st.get("interval_window") is not None and st.get("event_onsets") is not None
    if locked and st["units"] == "samples" and fs is None:
        logger.warning("segments made from intervals in sample indices without a "
                       "sampling rate: kept as Intervals")
        locked = False
    if not locked:
        keys = ("intervals", "units", "label", "event_labels", "event_onsets", "event_meta")
        return {"class": "Intervals", **{k: st.get(k) for k in keys}}
    if st["units"] == "samples":
        logger.warning("segments made from intervals in sample indices: event onsets "
                       "converted to ms from the first sample of the recording")
        fac = 1000. / fs
    else:
        fac = _V2_MS_PER_UNIT[st["units"]]
    return {"class": "Epochs", "onsets": np.asarray(st["event_onsets"], dtype=float) * fac,
            "window": tuple(float(w) * fac for w in st["interval_window"]),
            "labels": st["event_labels"], "meta": st.get("event_meta"),
            "label": st.get("label")}


def _v2_fix_fs(state):
    """Correct the sampling rate of a version-2 eyedata state (in place) if it
    does not fit the time vector (as after `downsample` before version 3, which
    stored the requested rate instead of the actual one)."""
    tx = np.asarray(state["tx"], dtype=float)
    fs = state.get("fs")
    if len(tx) < 2 or not isinstance(fs, (int, float, np.integer, np.floating)) or fs <= 0:
        return
    steps = (np.median(np.diff(tx)), (tx[-1] - tx[0]) / (len(tx) - 1))
    if any(np.isclose(step, 1000. / fs, rtol=0.05, atol=0) for step in steps) \
            or not steps[0] > 0:
        return
    new = 1000. / float(steps[0])
    logger.warning(f"{state.get('name')}: the sampling rate ({fs:g} Hz) does not match "
                   f"the time vector ({new:g} Hz, e.g. after downsample() in an older "
                   f"version of pypillometry); set to {new:g} Hz")
    state["fs"] = new


def _v2_eyedata_layers(state):
    """Move the masks, blinks, interpolated blinks and excluded periods of a
    version-2 eyedata state into the layers of its data (in place)."""
    had_layers = "layers" in state["data"]
    had_rec_layers = "rec_layers" in state["data"]
    state["data"], layers = _v2_eyedatadict(state["data"])
    values = state["data"]["values"]
    tx = np.asarray(state["tx"], dtype=float)
    blinks = state.pop("blinks", None) or {}
    interpolated = state.pop("interpolated_blinks", None) or {}
    excluded = state.pop("excluded", None)
    if not had_layers:
        # the mask of version 2 (all in the user layer so far)
        v2_mask = {key: layers.mask(key) for key in values}
        # blinks mask the data (also those that were not applied as mask in
        # version 2); the blink layer replaces the user layer there
        for key, ivst in blinks.items():
            if ivst is not None and key in values:
                blink = _v2_intervals_mask(ivst, tx)
                layers.set(key, "blink", blink)
                layers.clear(key, "user", blink)
        # interpolated blinks: only where they were (still) masked
        for key, ivst in interpolated.items():
            if ivst is not None and key in values:
                interp = _v2_intervals_mask(ivst, tx) & v2_mask[key]
                layers.set(key, "interpolated", interp)
                layers.clear(key, "user", interp)
    if not had_rec_layers and excluded is not None:
        excl = _v2_intervals_mask(excluded, tx)
        if excl.any():
            layers.set(None, "excluded", excl)
            for key in values:
                layers.clear(key, "user", excl)
    if state.get("sim_data") is not None:  # FakeEyeData
        state["sim_data"] = _v2_eyedatadict(state["sim_data"])[0]


def _v2_params(state):
    """Key the response parameters of a version-2 eyedata state by channel
    (``params["response"]["left"]`` -> ``["left_pupil"]``; replaces
    `state["params"]` by a new dict)."""
    params = state.get("params")
    if isinstance(params, dict) and isinstance(params.get("response"), dict):
        response = {(k if "_" in k else f"{k}_pupil"): v
                    for k, v in params["response"].items()}
        state["params"] = {**params, "response": response}


def _migrate_2_to_3(state, class_name):
    """Version 2 -> 3: the events of eyedata objects are one Events state
    (`events`) instead of `event_onsets`, `event_labels` and `event_meta`;
    masks are layers (see _v2_eyedata_layers); segments store Epochs (see
    _v2_spans); a sampling rate that does not fit the time vector is
    corrected (see _v2_fix_fs); all eyedata objects have an experimental
    setup and calibration (None for PupilData); the response parameters are
    keyed by channel (see _v2_params)."""
    state = dict(state)
    if class_name in _SEGMENTED_CLASSES:
        state["intervals"] = _v2_spans(state.get("intervals"), state.get("sampling_rate"))
        return state
    state["events"] = {"onsets": state.pop("event_onsets"),
                       "labels": state.pop("event_labels"), "units": "ms",
                       "data_time_range": None, "meta": state.pop("event_meta")}
    _v2_fix_fs(state)
    _v2_eyedata_layers(state)
    _v2_params(state)
    # PupilData had no experimental setup and calibration
    state.setdefault("experimental_setup", None)
    state.setdefault("calibration", None)
    return state


_MIGRATIONS = [_migrate_0_to_1, _migrate_1_to_2, _migrate_2_to_3]
assert len(_MIGRATIONS) == FORMAT_VERSION
