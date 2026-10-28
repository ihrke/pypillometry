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
    "pypillometry.cli": {"pypilview_main"},
    "pypillometry.convenience": {
        "change_dir", "suppress_all_output", "sizeof_fmt", "ByteSize",
        "test_selector", "is_url", "requires_package", "normalize_unit",
        "nprange", "p_asym_laplac", "p_asym_laplac_kappa", "trans_logistic_vec",
    },
    "pypillometry.example_data": {"DatasetDict"},
    "pypillometry.intervals": {"stat_event_interval"},
    "pypillometry.io": {"is_ssh_url"},
    "pypillometry.plot.genplotter": {"GenericPlotter"},
    "pypillometry.signal.preproc": {"helper_merge_blinks"},
    "pypillometry.viewer.canvas": {"ViewerCanvas"},
    "pypillometry.viewer.navigation": {"NavigationHandler"},
    "pypillometry.viewer.tweak": {"TweakCanvas", "TweakViewer"},
    "pypillometry.viewer.visuals": {
        "LODLine", "DynamicMaskRegions", "DynamicEventMarkers",
        "SelectionRegion", "HighlightRegion",
    },
}

# Modules that are not imported by the package (legacy code).
SKIP_MODULES = {"pypillometry.eyedata.fakedata"}


def _package_modules():
    for info in pkgutil.walk_packages(pp.__path__, "pypillometry."):
        if info.name not in SKIP_MODULES:
            yield importlib.import_module(info.name)


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
            for name in getattr(module, "__all__", []):
                with self.subTest(module=module.__name__, name=name):
                    self.assertIn(name, pp.__all__,
                                  "add it to pypillometry/__init__.py")

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
            for name in _public_definitions(module):
                with self.subTest(module=module.__name__, name=name):
                    self.assertTrue(
                        name in pp.__all__ or name in internal,
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


if __name__ == "__main__":
    unittest.main()
