"""
namespace.py
============

Base class of the method namespaces of :class:`~pypillometry.EyeData`
(``d.pupil``, ``d.gaze``) and the descriptor that attaches them to the class.
"""


class Namespace:
    """
    Base class of the method namespaces of :class:`~pypillometry.EyeData`.

    A namespace (e.g., ``d.pupil``) is a view on one data object: it holds
    only a reference to the object (no state of its own) and is created anew
    on each access, so it does not affect copying or saving. Its methods
    work on the data object and return the data object, so calls can be
    chained.

    Without data (``pp.pupil``, ``pp.gaze``), the transforming methods
    return pipeline steps instead of running (see :class:`~pypillometry.Step`).

    Parameters
    ----------
    data : EyeData or None
        the data object the methods work on, or `None` (pipeline steps only)
    """

    #: name of the namespace (attribute name on the data object)
    _name = None

    def __init__(self, data=None):
        self._data = data

    @property
    def _d(self):
        """The data object; raises a TypeError if there is none."""
        if self._data is None:
            raise TypeError(
                f"pp.{self._name} has no data: only the transforming methods "
                f"make pipeline steps (e.g., d | pp.{self._name}.<method>(...)); "
                f"call the other methods on the data: d.{self._name}.<method>(...)")
        return self._data

    @classmethod
    def _methods(cls):
        """Names of the public methods of the namespace."""
        return sorted(name for name in dir(cls)
                      if not name.startswith("_") and callable(getattr(cls, name)))

    def __repr__(self):
        methods = ", ".join(self._methods())
        if self._data is None:
            return f"<pp.{self._name}: {methods}>"
        return f"<{self._name} methods of {self._data.name!r}: {methods}>"


class accessor:
    """
    Descriptor that attaches a :class:`Namespace` to a data class.

    On the class (``EyeData.pupil``), it returns the namespace class (for
    documentation and tab completion); on an object (``d.pupil``), a new
    namespace bound to the object.

    Parameters
    ----------
    cls : type
        the namespace class (a subclass of :class:`Namespace`)
    """

    def __init__(self, cls):
        self.cls = cls
        self.__doc__ = cls.__doc__

    def __get__(self, obj, owner=None):
        return self.cls if obj is None else self.cls(obj)
