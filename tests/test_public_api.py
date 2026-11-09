"""Guard tests for the package's public API.

Each submodule lists its public names in ``__all__``, and
``pypillometry/__init__.py`` imports and lists them explicitly. These tests
fail when the two get out of sync, or when a new public function or class is
neither exported nor marked as internal.
"""
import importlib
import inspect
import pkgutil
import types
import unittest

import pypillometry as pp


# Public-looking definitions that are intentionally not exported at the top
# level. They stay importable from their submodule.
INTERNAL = {
    "pypillometry.cli": {"pypilview_main", "edf2pypil_main", "pypilinfo_main",
                         "pypil_upgrade_main"},
    "pypillometry.convenience": {
        "change_dir", "suppress_all_output", "sizeof_fmt", "ByteSize",
        "test_selector", "is_url", "requires_package", "normalize_unit",
    },
    "pypillometry.example_data": {"DatasetDict"},
    # namespaces: used as d.pupil / pp.pupil, documented in their modules
    "pypillometry.eyedata.namespace": {"Namespace", "accessor"},
    "pypillometry.eyedata.pupil": {"PupilNamespace"},
    "pypillometry.eyedata.gaze": {"GazeNamespace"},
    "pypillometry.eyedata.pipes": {"step", "generic_step"},
    "pypillometry.intervals": {"stat_event_interval"},
    "pypillometry.io": {"is_ssh_url"},
    "pypillometry.plot": {"PlotNamespace"},
    "pypillometry.segmented": {"SegmentedPupilNamespace", "SegmentedGazeNamespace", "sem"},
    "pypillometry.plot.generic": {"GenericPlots"},
    "pypillometry.plot.pupil": {"PupilPlots"},
    "pypillometry.plot.gaze": {"GazePlots"},
    "pypillometry.units": {"normalize_unit"},
    "pypillometry.signal.preproc": {"helper_merge_blinks"},
    "pypillometry.templates": {"render", "sparkline"},
    "pypillometry.state": {"to_state", "from_state", "migrate", "Field",
                           "Stateful", "VersionedState"},
    "pypillometry.hdf5": {"write_state", "read_state", "is_hdf5"},
    "pypillometry.viewer.canvas": {"ViewerCanvas"},
    "pypillometry.viewer.navigation": {"NavigationHandler"},
    "pypillometry.viewer.tweak_canvas": {"TweakCanvas", "TweakViewer"},
    "pypillometry.viewer.visuals": {
        "LODLine", "DynamicMaskRegions", "DynamicEventMarkers",
        "SelectionRegion", "HighlightRegion",
    },
}

# Names in a submodule's __all__ that the top level uses for something else:
# the low-level function stays in its submodule (e.g. pp.signal.downsample),
# pp.<name> is the pipeline step.
SHADOWED = {
    "pypillometry.signal": {"downsample"},
    "pypillometry.signal.baseline": {"downsample"},
}

# Modules whose public names are only in a namespace submodule, not at the
# top level (pp.signal.smooth_window, not pp.smooth_window).
SUBMODULE_API = {"pypillometry.signal": "pypillometry.signal"}

# Modules that are not imported by the package (legacy code).
SKIP_MODULES = set()


def _package_modules():
    for info in pkgutil.walk_packages(pp.__path__, "pypillometry."):
        if info.name not in SKIP_MODULES:
            yield importlib.import_module(info.name)


def _api_module(module):
    """The module whose __all__ exports the public names of `module`."""
    for prefix, api in SUBMODULE_API.items():
        if module.__name__ == prefix or module.__name__.startswith(prefix + "."):
            return importlib.import_module(api)
    return pp


def _public_definitions(module):
    """Public functions and classes defined in (not imported into) module."""
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if obj.__module__ == module.__name__:
            yield name


class TestPublicAPI(unittest.TestCase):

    def test_all_names_exist(self):
        for module in _package_modules():
            for name in getattr(module, "__all__", []):
                with self.subTest(module=module.__name__, name=name):
                    self.assertTrue(hasattr(module, name))

    def test_submodule_all_is_exported(self):
        for module in _package_modules():
            shadowed = SHADOWED.get(module.__name__, set())
            api = _api_module(module)
            for name in getattr(module, "__all__", []):
                with self.subTest(module=module.__name__, name=name):
                    if api is not pp:
                        self.assertIn(name, api.__all__)
                        if name not in shadowed:
                            self.assertNotIn(name, pp.__all__,
                                             f"{name} belongs to {api.__name__} only")
                        continue
                    self.assertIn(name, pp.__all__,
                                  "add it to pypillometry/__init__.py")
                    if name in shadowed:
                        self.assertIsNot(getattr(pp, name), getattr(module, name))
                    else:
                        self.assertIs(getattr(pp, name), getattr(module, name),
                                      "same name, different object: add to SHADOWED")

    def test_viewer_tweak_is_the_function(self):
        # the submodule viewer/tweak.py shadowed the function once imported
        import pypillometry.viewer as viewer
        import pypillometry.viewer.tweak_canvas  # noqa: F401
        self.assertTrue(callable(viewer.tweak))
        self.assertIs(viewer.tweak, pp.tweak)

    def test_top_level_all_is_clean(self):
        self.assertEqual(len(pp.__all__), len(set(pp.__all__)),
                         "duplicate names in pypillometry.__all__")
        for name in pp.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pp, name))
                obj = getattr(pp, name)
                self.assertNotIsInstance(obj, types.ModuleType)

    def test_new_definitions_are_exported_or_internal(self):
        for module in _package_modules():
            internal = INTERNAL.get(module.__name__, set())
            api = _api_module(module)
            for name in _public_definitions(module):
                with self.subTest(module=module.__name__, name=name):
                    self.assertTrue(
                        name in api.__all__ or name in internal,
                        f"{module.__name__}.{name} is public but neither in "
                        f"__all__ nor listed in INTERNAL ({__file__})",
                    )

    def test_internal_list_is_current(self):
        for modname, names in INTERNAL.items():
            module = importlib.import_module(modname)
            for name in names:
                with self.subTest(module=modname, name=name):
                    self.assertTrue(hasattr(module, name),
                                    "stale entry in INTERNAL")
                    self.assertNotIn(name, pp.__all__,
                                     "exported name listed in INTERNAL")

    def test_submodules_not_shadowed(self):
        # Third-party modules with the same name must not replace
        # the package's own submodules (e.g. scipy.signal).
        for name in ["signal", "io", "logging", "plot", "eyedata", "viewer"]:
            with self.subTest(name=name):
                self.assertEqual(getattr(pp, name).__name__,
                                 f"pypillometry.{name}")

    def test_signal_functions_only_in_signal(self):
        self.assertTrue(callable(pp.signal.smooth_window))
        self.assertFalse(hasattr(pp, "smooth_window"))

    def test_generic_eyedata_is_eyedata(self):
        self.assertFalse(hasattr(pp, "GenericEyeData"))
        self.assertEqual(pp.EyeData.__bases__, (pp.state.VersionedState,))


if __name__ == "__main__":
    unittest.main()
