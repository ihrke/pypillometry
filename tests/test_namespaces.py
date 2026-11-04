"""Tests for the method namespaces (``d.pupil``, ``d.gaze``)."""
import numpy as np
import pytest

import pypillometry as pp
from pypillometry.eyedata.namespace import Namespace, accessor
from pypillometry.eyedata.generic import keephistory


class _Demo(Namespace):
    """Demo namespace."""
    _name = "demo"

    @keephistory
    def scale_pupil(self, factor, inplace=None):
        obj = self._d._get_inplace(inplace)
        obj.data["left_pupil"] = obj.data["left_pupil"] * factor
        return obj

    def first(self):
        return self._d.data["left_pupil"][0]


class _DemoData(pp.EyeData):
    demo = accessor(_Demo)


def _data():
    return _DemoData(left_pupil=np.ones(100), sampling_rate=100, name="demo")


class TestNamespaceBase:
    def test_class_access_gives_namespace_class(self):
        assert _DemoData.demo is _Demo

    def test_object_access_gives_view(self):
        d = _data()
        assert isinstance(d.demo, _Demo)
        assert d.demo._d is d
        assert d.demo is not d.demo  # created on access, never stored
        assert "demo" not in vars(d)

    def test_method_returns_data_and_chains(self):
        d = _data()
        res = d.demo.scale_pupil(2).demo.scale_pupil(3)
        assert res is d
        assert d.demo.first() == 6

    def test_history_has_namespace_name(self):
        d = _data()
        d.demo.scale_pupil(2)
        assert d.history[-1]["funcname"] == "demo.scale_pupil"
        assert d.history[-1]["funcstring"] == "demo.scale_pupil(2)"

    def test_inplace_false(self):
        d = _data()
        res = d.demo.scale_pupil(2, inplace=False)
        assert res is not d
        assert d.demo.first() == 1 and res.demo.first() == 2
        assert d.history == [] or d.history[-1]["funcname"] != "demo.scale_pupil"

    def test_without_data(self):
        ns = _Demo()
        with pytest.raises(TypeError, match="has no data"):
            ns.first()

    def test_repr(self):
        assert repr(_Demo()) == "<pp.demo: first, scale_pupil>"
        assert repr(_data().demo) == "<demo methods of 'demo': first, scale_pupil>"

    def test_copy_and_pickle_unaffected(self, tmp_path):
        d = _data()
        d.copy().demo.scale_pupil(2)
        fname = str(tmp_path / "d.pkl")
        pp.write_pickle(d, fname)
        assert pp.read_pickle(fname).demo.first() == 1


class TestRenamedMethods:
    """The flat pupil/gaze names of 2.1 are gone; the error names the new one."""

    def test_rename_table(self):
        from pypillometry.eyedata.generic import _RENAMED_METHODS
        d = pp.EyeData(left_pupil=np.ones(10), left_x=np.ones(10), left_y=np.ones(10),
                       sampling_rate=10)
        for old, new in _RENAMED_METHODS.items():
            ns, name = new.split(".")
            assert callable(getattr(getattr(d, ns), name)), new
            assert not hasattr(pp.EyeData, old), old
            with pytest.raises(AttributeError, match=rf"renamed: use \.{ns}\.{name}\(\.\.\.\)"):
                getattr(d, old)

    def test_flat_names_gone(self):
        flat = [n for n in dir(pp.EyeData)
                if n.startswith(("pupil_", "gaze_", "mask_offscreen", "mask_eye"))]
        assert flat == []
