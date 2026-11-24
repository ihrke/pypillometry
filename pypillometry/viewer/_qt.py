"""
The Qt binding of the viewer: the one that is already imported, or else the
first one that is installed (PyQt6, PySide6, PyQt5, PySide2). vispy is set to
the same one (see :func:`use_vispy`).
"""
import importlib
import sys

_BINDINGS = ("PyQt6", "PySide6", "PyQt5", "PySide2")
_binding = None


def binding() -> str:
    """name of the Qt binding (e.g. ``"PySide6"``); raises an ImportError if
    none is installed"""
    global _binding
    if _binding is None:
        imported = [b for b in _BINDINGS if b in sys.modules or f"{b}.QtCore" in sys.modules]
        errors = []
        for name in imported + [b for b in _BINDINGS if b not in imported]:
            try:
                importlib.import_module(f"{name}.QtWidgets")
            except ImportError as e:
                if not isinstance(e, ModuleNotFoundError) or e.name not in (name, None):
                    errors.append(f"{name}: {e}")      # installed, but does not load
                continue
            _binding = name
            break
        else:
            msg = f"the viewer needs Qt: install one of {', '.join(_BINDINGS)} (e.g., PySide6)"
            if errors:
                msg += "; installed, but cannot be loaded: " + "; ".join(errors)
            raise ImportError(msg)
    return _binding


def qt(module: str):
    """the Qt module `module` (e.g. ``"QtWidgets"``) of the binding"""
    return importlib.import_module(f"{binding()}.{module}")


def use_vispy():
    """set vispy to the binding"""
    import vispy
    vispy.use(app=binding().lower())
