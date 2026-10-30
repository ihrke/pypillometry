"""
Versioned state of the objects that can be saved to a file.

The eyedata objects (:class:`~pypillometry.PupilData`,
:class:`~pypillometry.GazeData`, :class:`~pypillometry.EyeData`, ...) and the
segmented data (:class:`~pypillometry.SegmentedEyeData`, ...) define their
content as a *state*: a dict that contains only numbers, strings, None,
lists, tuples, dicts and numpy arrays (plus whatever the user stored in
`info` and `params`). Its keys ``format_version`` and ``class`` say which
version of the layout the state has and which class it belongs to.

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
    the state as defined by the ``_to_state()`` methods of the classes
"""

import importlib
import pickle

import numpy as np
from loguru import logger

__all__ = []

#: version of the state written by this version of pypillometry
FORMAT_VERSION = 1

# classes that moved: (old module, old name) -> (new module, new name),
# used when reading pickle files (see _Unpickler)
_RENAMED_CLASSES = {}


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


def _find_class(name: str):
    """The class (eyedata or segmented data) called `name`."""
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
# Migrations: _MIGRATIONS[v] upgrades a state from version v to v + 1.
# These functions are frozen: they describe the layout at that version and
# must not change when the classes change later.
# --------------------------------------------------------------------------

_EYEDATA_CLASSES = ("PupilData", "GazeData", "EyeData", "FakeEyeData")
_SEGMENTED_CLASSES = ("SegmentedEyeData", "GroupLevelSegmentedData")


def _v0_dict(obj):
    """Raw attributes of an object from a version-0 pickle (or None)."""
    return None if obj is None else vars(obj)


def _v0_eyedatadict(raw):
    """Version-1 state of an EyeDataDict (raw attributes `raw`)."""
    return {"data": dict(raw["data"]), "mask": dict(raw["mask"])}


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
            "data": np.ma.getdata(data),
            "mask": np.ma.getmaskarray(data),
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
        **_v0_eyedatadict(vars(raw["data"])),
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


_MIGRATIONS = [_migrate_0_to_1]
assert len(_MIGRATIONS) == FORMAT_VERSION
