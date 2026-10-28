"""Units for times, distances and angles.

Values with units can be given in two ways:

- multiplied by a unit from the :data:`u` namespace: ``3 * u.min``,
  ``(-200, 800) * u.ms``, ``60 * u.cm``, ``20 * u.deg``
- as strings: ``"3 min"``, ``"60 cm"``, ``"20 deg"``

Internally, pypillometry converts everything to canonical units and stores
plain floats or arrays: ms for times, mm for distances and radians for
angles. Each function that takes such a value documents what a plain number
(without units) means.

Examples
--------
>>> from pypillometry import u
>>> (3 * u.min).to(u.sec)
180.0 sec
>>> (-200, 800) * u.ms
[-200.  800.] ms
>>> parse_time("2 sec")
2000.0
>>> parse_distance("60 cm")
600.0
"""

import inspect
import math
import re
import warnings

import numpy as np

__all__ = [
    "u", "Unit", "Quantity",
    "parse_time", "parse_distance", "parse_angle",
]


class Unit:
    """A unit of time, length or angle.

    Multiplying a number, tuple, list or array by a unit gives a
    :class:`Quantity`. Units are available in the :data:`u` namespace.

    Parameters
    ----------
    name : str
        canonical name of the unit (e.g., "ms", "sec", "mm", "deg")
    dimension : str
        one of "time", "length", "angle"
    factor : float or None
        size of the unit in the base unit of its dimension (ms, mm or radians);
        None for "samples", whose size depends on the sampling rate
    """
    __slots__ = ("name", "dimension", "factor")
    __array_ufunc__ = None  # numpy arrays defer `array * unit` to __rmul__

    def __init__(self, name, dimension, factor):
        self.name = name
        self.dimension = dimension
        self.factor = factor

    def __rmul__(self, value):
        return Quantity(value, self)

    __mul__ = __rmul__

    def __repr__(self):
        return f"<Unit {self.name}>"

    def __str__(self):
        return self.name

    def __reduce__(self):
        # unpickle to the shared instance
        return (_get_unit, (self.name, self.dimension))


class Quantity:
    """A number or array together with a :class:`Unit`.

    Usually created by multiplying with a unit, e.g. ``3 * u.min`` or
    ``(-200, 800) * u.ms``. Tuples and lists are stored as numpy arrays.
    Iterating over or indexing an array quantity gives quantities.

    Parameters
    ----------
    value : float or array-like
        magnitude
    unit : Unit or str
        unit of `value`
    """
    __slots__ = ("value", "unit")
    __array_ufunc__ = None  # keep numpy from treating quantities as objects

    def __init__(self, value, unit):
        if isinstance(value, (list, tuple, np.ndarray)):
            value = np.asarray(value, dtype=float)
        elif isinstance(value, Quantity):
            raise TypeError("cannot multiply a Quantity by a Unit; use .to()")
        self.value = value
        self.unit = _get_unit(unit) if isinstance(unit, str) else unit

    @property
    def magnitude(self):
        """The value without unit."""
        return self.value

    @property
    def dimension(self):
        """Dimension of the unit ("time", "length" or "angle")."""
        return self.unit.dimension

    def to(self, unit, fs=None):
        """Return the quantity converted to `unit`.

        Parameters
        ----------
        unit : Unit or str
            target unit; must have the same dimension
        fs : float, optional
            sampling rate in Hz; needed to convert from or to samples
        """
        unit = _get_unit(unit, self.dimension)
        return Quantity(_convert(self.value, self.unit, unit, fs), unit)

    def __len__(self):
        return len(self.value)

    def __bool__(self):
        return bool(np.any(self.value))

    def __iter__(self):
        return (Quantity(v, self.unit) for v in np.asarray(self.value))

    def __getitem__(self, idx):
        return Quantity(self.value[idx], self.unit)

    def __neg__(self):
        return Quantity(-self.value, self.unit)

    def __add__(self, other):
        if not isinstance(other, Quantity):
            return NotImplemented
        return Quantity(self.value + other.to(self.unit).value, self.unit)

    def __sub__(self, other):
        if not isinstance(other, Quantity):
            return NotImplemented
        return Quantity(self.value - other.to(self.unit).value, self.unit)

    def __mul__(self, other):
        if isinstance(other, (Quantity, Unit)):
            return NotImplemented
        return Quantity(self.value * other, self.unit)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (Quantity, Unit)):
            return NotImplemented
        return Quantity(self.value / other, self.unit)

    def __eq__(self, other):
        if not isinstance(other, Quantity) or other.dimension != self.dimension:
            return NotImplemented
        return bool(np.all(self.value == other.to(self.unit).value))

    __hash__ = None

    def __repr__(self):
        return f"{self.value} {self.unit.name}"


# Units by dimension: canonical name, size in base unit, aliases.
# The base units are ms, mm and radians.
_UNIT_TABLE = {
    "time": [
        ("ms", 1.0, ["millisecond", "milliseconds", "msec"]),
        ("sec", 1000.0, ["s", "secs", "second", "seconds"]),
        ("min", 60000.0, ["m", "mins", "minute", "minutes"]),
        ("h", 3600000.0, ["hr", "hrs", "hour", "hours"]),
        ("samples", None, ["sample", "index", "indices", "idx"]),
    ],
    "length": [
        ("mm", 1.0, ["millimeter", "millimeters", "millimetre", "millimetres"]),
        ("cm", 10.0, ["centimeter", "centimeters", "centimetre", "centimetres"]),
        ("m", 1000.0, ["meter", "meters", "metre", "metres"]),
        ("inch", 25.4, ["in", "inches"]),
    ],
    "angle": [
        ("rad", 1.0, ["radian", "radians"]),
        ("deg", math.pi / 180.0, ["degree", "degrees", "°"]),
    ],
}

# name/alias -> Unit, separately for each dimension
_UNITS = {}
for _dim, _entries in _UNIT_TABLE.items():
    _UNITS[_dim] = {}
    for _name, _factor, _aliases in _entries:
        _unit = Unit(_name, _dim, _factor)
        for _alias in [_name] + _aliases:
            _UNITS[_dim][_alias] = _unit

# without a dimension, "m" means metre (it means minutes for times)
_DIMENSION_ORDER = ("length", "time", "angle")


def _get_unit(unit, dimension=None):
    """Return the Unit for a name/alias or Unit, optionally checking its dimension."""
    if isinstance(unit, Unit):
        if dimension is not None and unit.dimension != dimension:
            raise ValueError(f"expected a unit of {dimension}, got '{unit.name}' "
                             f"({unit.dimension})")
        return unit
    key = str(unit).strip()
    dims = (dimension,) if dimension is not None else _DIMENSION_ORDER
    for dim in dims:
        for k in (key, key.lower()):
            if k in _UNITS[dim]:
                return _UNITS[dim][k]
    if dimension is not None:
        for dim in _DIMENSION_ORDER:
            if key.lower() in _UNITS[dim]:
                raise ValueError(f"expected a unit of {dimension}, got '{unit}' ({dim})")
        valid = ", ".join(n for n, _, _ in _UNIT_TABLE[dimension])
        raise ValueError(f"Unknown unit '{unit}'. Valid units are: {valid}")
    raise ValueError(f"Unknown unit '{unit}'")


def _convert(value, from_unit, to_unit, fs=None):
    """Convert a magnitude between two units of the same dimension."""
    if from_unit is to_unit:
        return value
    if from_unit.factor is None or to_unit.factor is None:
        if fs is None:
            raise ValueError(f"converting between '{from_unit.name}' and "
                             f"'{to_unit.name}' needs the sampling rate")
        ms_per_sample = 1000.0 / fs
    from_factor = from_unit.factor if from_unit.factor is not None else ms_per_sample
    to_factor = to_unit.factor if to_unit.factor is not None else ms_per_sample
    return value * (from_factor / to_factor)


class _UnitNamespace:
    """Namespace of units, e.g. ``u.ms``, ``u.sec``, ``u.min``, ``u.cm``, ``u.deg``.

    All aliases work as attributes (e.g. ``u.seconds``, ``u.degrees``).
    ``u.m`` is metres (use ``u.min`` for minutes).
    """

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return _get_unit(name)
        except ValueError:
            raise AttributeError(f"unknown unit '{name}'") from None

    def __dir__(self):
        return sorted({a for d in _UNITS.values() for a in d if a.isidentifier()})

    def __repr__(self):
        return "<units: " + ", ".join(n for d in _UNIT_TABLE.values() for n, _, _ in d) + ">"


u = _UnitNamespace()
"""Namespace of units: ``u.ms``, ``u.sec``, ``u.min``, ``u.h``, ``u.samples``,
``u.mm``, ``u.cm``, ``u.m``, ``u.inch``, ``u.deg``, ``u.rad`` (and aliases)."""


_NUMBER_UNIT_RE = re.compile(
    r"^\s*([-+]?(?:inf|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))\s*(\S.*?)?\s*$",
    re.IGNORECASE)


def _split(value, dimension):
    """Split `value` into (magnitude, Unit or None); None means no unit given."""
    if isinstance(value, Quantity):
        _get_unit(value.unit, dimension)  # check dimension
        return value.value, value.unit
    if isinstance(value, str):
        match = _NUMBER_UNIT_RE.match(value)
        if match is None:
            raise ValueError(f"Invalid unit string '{value}'; expected a number with unit "
                             f"such as '3 min', '60 cm' or '20 deg'")
        number, unit = match.groups()
        return float(number), (_get_unit(unit, dimension) if unit else None)
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        # pint-like quantity (pint itself is not needed)
        return np.asarray(value.magnitude, dtype=float) if np.ndim(value.magnitude) \
            else float(value.magnitude), _get_unit(str(value.units), dimension)
    return value, None


def _is_sequence(value):
    return isinstance(value, (list, tuple)) or (
        isinstance(value, np.ndarray) and value.dtype == object)


def _parse(value, dimension, to, default, fs=None):
    """Convert `value` to unit `to`; values without unit are in `default`.

    Returns None, a float or an ndarray (for sequences).
    """
    if value is None:
        return None
    to = _get_unit(to, dimension)
    if _is_sequence(value):
        return np.array([_parse(v, dimension, to, default, fs) for v in value], dtype=float)
    magnitude, unit = _split(value, dimension)
    if unit is None:
        if default is None:
            raise ValueError(f"'{value}' has no unit")
        unit = _get_unit(default, dimension)
    if isinstance(magnitude, (list, tuple, np.ndarray)):
        magnitude = np.asarray(magnitude, dtype=float)
    else:
        magnitude = float(magnitude)
    return _convert(magnitude, unit, to, fs)


def _has_units(value):
    """True if `value` (or any element of a tuple/list) carries a unit."""
    if _is_sequence(value):
        return any(_has_units(v) for v in value)
    if isinstance(value, str):
        return _NUMBER_UNIT_RE.match(value) is not None and \
            _NUMBER_UNIT_RE.match(value).group(2) is not None
    return isinstance(value, Quantity) or (
        hasattr(value, "magnitude") and hasattr(value, "units"))


def _first_unit(value, dimension):
    """Unit of the first element of `value` that carries one (or None)."""
    if _is_sequence(value):
        for v in value:
            unit = _first_unit(v, dimension)
            if unit is not None:
                return unit
        return None
    return _split(value, dimension)[1] if _has_units(value) else None


UNSET = object()
"""Default for `units=` arguments of containers: take the unit from the values."""


def _time_values_and_units(value, units=UNSET, fs=None):
    """Values and canonical unit name for containers such as Intervals.

    If `value` carries units, it is converted to `units` (a time unit, or
    None for samples) or, if `units` is UNSET, to the unit of its first
    element. Plain values are returned unchanged, together with `units`.
    """
    if not _has_units(value):
        if units is UNSET:
            raise ValueError("plain numbers need units (e.g., units='ms'); "
                             "or give values with units, e.g., (0, 100) * u.ms")
        return value, normalize_unit(units)
    if units is UNSET:
        unit = _first_unit(value, "time")
    else:
        unit = _get_unit("samples" if units is None else units, "time")
    values = _parse(value, "time", unit, None, fs)
    return values, normalize_unit(unit)


def parse_time(value, to="ms", default="ms", fs=None):
    """Convert a time (or tuple/array of times) to unit `to`.

    Parameters
    ----------
    value : float, array-like, str, Quantity, tuple or None
        time value(s), e.g. ``500``, ``"2 sec"``, ``3 * u.min``,
        ``(-200, 800) * u.ms`` or ``("1 min", "2 min")``
    to : str or Unit, optional
        target unit; default "ms"
    default : str, Unit or None, optional
        unit of plain numbers; default "ms". If None, plain numbers raise
        a ValueError.
    fs : float, optional
        sampling rate in Hz; needed to convert from or to "samples"

    Returns
    -------
    float, ndarray or None
        the value in unit `to` (an ndarray for tuples, lists and arrays)

    Examples
    --------
    >>> parse_time("2 sec")
    2000.0
    >>> parse_time(3 * u.min, to="sec")
    180.0
    >>> parse_time((-200, 800), to="sec")
    array([-0.2,  0.8])
    >>> parse_time(10 * u.samples, fs=500)
    20.0
    """
    return _parse(value, "time", to, default, fs)


# Session-wide tracking of issued warnings (one per parameter name)
_unit_warnings_issued = set()


def _get_param_name():
    """Name of the variable that the caller of parse_* assigns to.

    For ``camera_eye_distance = parse_distance(value)``, this returns
    "camera_eye_distance"; "parameter" if it cannot be determined.
    """
    try:
        # Go back two frames: _get_param_name -> parse_* -> caller
        frame = inspect.currentframe().f_back.f_back
        code_context = inspect.getframeinfo(frame).code_context
        if code_context:
            line = code_context[0].strip()
            if '=' in line and '==' not in line.split('=')[0]:
                return line.split('=')[0].strip()
    except Exception:
        pass
    return "parameter"


def _warn_plain(value, unit, unit_name):
    param_name = _get_param_name()
    if param_name not in _unit_warnings_issued:
        warnings.warn(
            f"{param_name}: plain number provided, assuming {unit_name}. "
            f"Use a string ('{value} {unit}') or a unit (value * u.{unit}) "
            f"for explicit units.",
            UserWarning,
            stacklevel=3
        )
        _unit_warnings_issued.add(param_name)


def parse_distance(value):
    """Convert a distance to mm.

    Parameters
    ----------
    value : float, str, Quantity or None
        e.g. ``"60 cm"``, ``60 * u.cm``, ``"0.6 m"``; plain numbers are mm
        (with a one-time warning)

    Returns
    -------
    float, ndarray or None
        distance in mm

    Examples
    --------
    >>> parse_distance("60 cm")
    600.0
    >>> parse_distance(60 * u.cm)
    600.0
    """
    if value is None:
        return None
    if not _has_units(value):
        _warn_plain(value, "mm", "mm")
    return _parse(value, "length", "mm", "mm")


def parse_angle(value):
    """Convert an angle to radians.

    Parameters
    ----------
    value : float, str, Quantity or None
        e.g. ``"20 deg"``, ``20 * u.deg``, ``"0.349 rad"``; plain numbers are
        radians (with a one-time warning)

    Returns
    -------
    float, ndarray or None
        angle in radians

    Examples
    --------
    >>> round(parse_angle("180 deg"), 4)
    3.1416
    """
    if value is None:
        return None
    if not _has_units(value):
        _warn_plain(value, "rad", "radians")
    return _parse(value, "angle", "rad", "rad")


# canonical time unit names used by `units=` arguments ("samples" is None)
CANONICAL_UNITS = {"ms", "sec", "min", "h"}
UNIT_ALIASES = {alias: (unit.name if unit.factor is not None else None)
                for alias, unit in _UNITS["time"].items()
}


def normalize_unit(unit):
    """Normalize a time unit to its canonical name.

    Accepts aliases (e.g., "seconds", "s", "secs") and :class:`Unit` objects
    (e.g., ``u.sec``) and returns "ms", "sec", "min" or "h". Returns None if
    `unit` is None or means sample indices ("index", "indices", "idx",
    "sample", "samples").

    Parameters
    ----------
    unit : str, Unit or None
        unit to normalize; case and surrounding whitespace are ignored

    Returns
    -------
    str or None
        canonical unit name, or None for indices/samples (or None input)

    Raises
    ------
    ValueError
        If the unit is not a known time unit

    Examples
    --------
    >>> normalize_unit("seconds")
    'sec'
    >>> normalize_unit(u.min)
    'min'
    >>> normalize_unit("samples") is None
    True
    >>> normalize_unit("invalid")
    Traceback (most recent call last):
        ...
    ValueError: Unknown unit 'invalid'. Valid units are: h, min, ms, sec
    """
    if unit is None:
        return None
    if not isinstance(unit, Unit):
        key = str(unit).lower().strip()
        if key not in UNIT_ALIASES:
            valid = ", ".join(sorted(CANONICAL_UNITS))
            raise ValueError(f"Unknown unit '{unit}'. Valid units are: {valid}")
        return UNIT_ALIASES[key]
    unit = _get_unit(unit, "time")
    return unit.name if unit.factor is not None else None
