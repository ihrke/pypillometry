"""Package-level settings."""

__all__ = [
    "set_default_inplace", "get_default_inplace",
]

_default_inplace = True


def set_default_inplace(inplace: bool) -> None:
    """Set the package-level default for `inplace`.

    The setting is used by all methods called with `inplace=None` (the
    default) on objects that were created with `inplace=None` (the default).
    An `inplace` value given to the constructor or to a method overrides it.

    Parameters
    ----------
    inplace : bool
        if `True` (the initial default), methods change the object in place
        and return it; if `False`, they work on and return a copy

    Examples
    --------
    >>> import pypillometry as pp
    >>> pp.set_default_inplace(False)  # methods now return modified copies
    >>> pp.get_default_inplace()
    False
    >>> pp.set_default_inplace(True)
    """
    global _default_inplace
    if not isinstance(inplace, bool):
        raise TypeError(f"inplace must be True or False, got {inplace!r}")
    _default_inplace = inplace


def get_default_inplace() -> bool:
    """Return the package-level default for `inplace`.

    Returns
    -------
    bool
        the current default (see :func:`set_default_inplace`)
    """
    return _default_inplace
