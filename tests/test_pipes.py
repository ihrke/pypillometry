"""Tests for the pipeline steps (``d | pp.pupil.blinks_detect() | ...``)."""
import inspect

import numpy as np
import pytest

import pypillometry as pp
from pypillometry.eyedata import steps
from pypillometry import Step, Pipeline
from pypillometry.eyedata.namespace import Namespace


def _data(name="subj1"):
    rng = np.random.default_rng(1)
    n = 3000
    p = 5 + 0.1 * np.sin(np.arange(n) / 100) + 0.01 * rng.standard_normal(n)
    p[1000:1020] = 0  # a blink
    p[1030:1040] = 0  # a close one
    x = 640 + rng.standard_normal(n)
    y = 512 + rng.standard_normal(n)
    x[2000:2010] = 5000  # off-screen
    return pp.EyeData(left_pupil=p, left_x=x, left_y=y, sampling_rate=1000, name=name,
                      event_onsets=[500, 1500, 2500], event_labels=["a", "b", "a"],
                      experimental_setup=pp.ExperimentalSetup(screen_resolution=(1280, 1024)))


PREPROC = (pp.pupil.blinks_detect(strategies=["zero"])
           | pp.pupil.blinks_merge_close(distance=20)
           | pp.pupil.blinks_interpolate()
           | pp.pupil.lowpass_filter(5))


class TestSteps:
    def test_namespace_without_data_makes_steps(self):
        s = pp.pupil.lowpass_filter(5, order=3)
        assert isinstance(s, Step)
        assert repr(s) == "pp.pupil.lowpass_filter(5, order=3)"
        assert repr(pp.gaze.mask_offscreen()) == "pp.gaze.mask_offscreen()"
        assert repr(pp.crop("1 min", 30)) == "pp.crop('1 min', 30)"

    def test_same_implementation(self):
        """method call and step run the same code"""
        d = _data()
        a = d.pupil.blinks_detect(strategies=["zero"], inplace=False)
        b = d | pp.pupil.blinks_detect(strategies=["zero"])
        assert a.get_blinks(units="samples")["left_pupil"].intervals \
            == b.get_blinks(units="samples")["left_pupil"].intervals
        c = d.crop(100, 2000, inplace=False)
        e = d | pp.crop(100, 2000)
        np.testing.assert_array_equal(c.tx, e.tx)

    def test_factory_signature_and_doc(self):
        sig = inspect.signature(pp.crop)
        assert list(sig.parameters) == ["start", "end", "units"]
        assert pp.crop.__doc__.startswith("Pipeline step for :meth:`~pypillometry.EyeData.crop`")
        assert pp.crop.__doc__.endswith(inspect.cleandoc(pp.EyeData.crop.__doc__))
        assert pp.pupil.lowpass_filter.__doc__ == pp.EyeData.pupil.lowpass_filter.__doc__

    def test_arguments_checked_when_made(self):
        with pytest.raises(TypeError, match=r"pp.pupil.lowpass_filter\(\).*cutof"):
            pp.pupil.lowpass_filter(5, cutof=3)
        with pytest.raises(TypeError, match=r"pp.crop\(\).*inplace"):
            pp.crop(0, 100, inplace=False)
        with pytest.raises(TypeError, match="inplace"):
            pp.pupil.blinks_detect(inplace=True)

    def test_non_steps(self):
        """getters are not steps; pp.pupil has no data to run them on"""
        with pytest.raises(TypeError, match="pp.pupil has no data"):
            pp.pupil.fit_foreshortening("left")

    def test_step_marks(self):
        """the step factories exist exactly for the methods marked with @step"""
        marked = {n for n in dir(pp.EyeData)
                  if getattr(getattr(pp.EyeData, n, None), "_is_step", False)}
        assert marked == set(steps.__all__) - {"pupil", "gaze"}
        for ns in (pp.pupil, pp.gaze):
            for n in type(ns)._methods():
                if n != "fit_foreshortening":
                    assert getattr(type(ns), n)._is_step, n

    def test_step_methods_return_data_and_take_inplace(self):
        for n in set(steps.__all__) - {"pupil", "gaze"}:
            params = inspect.signature(getattr(pp.EyeData, n)).parameters
            assert ("inplace" in params) != n.startswith("set_"), n


class TestPipeline:
    def test_compose(self):
        p = pp.pupil.blinks_detect() | pp.pupil.lowpass_filter(5)
        assert isinstance(p, Pipeline) and len(p) == 2
        q = p | pp.downsample(50)
        assert len(p) == 2 and len(q) == 3  # values: p is unchanged
        assert len(p | q) == 5
        assert repr(q) == ("pp.pupil.blinks_detect() | pp.pupil.lowpass_filter(5)"
                           " | pp.downsample(50)")
        assert repr(Pipeline()) == "Pipeline([])"
        assert isinstance(q[1:], Pipeline) and len(q[1:]) == 2
        assert q[0] is p[0]

    def test_pipe_does_not_change_input(self):
        d = _data()
        before = (d.data["left_pupil"].copy(), len(d.history), d.data.mask["left_pupil"].sum())
        r = d | PREPROC | pp.downsample(100)
        assert r is not d
        np.testing.assert_array_equal(d.data["left_pupil"], before[0])
        assert len(d.history) == before[1]
        assert d.data.mask["left_pupil"].sum() == before[2]
        assert len(r) == len(d) // 10 and r.fs == 100
        assert r.name == d.name

    def test_copies_once(self, monkeypatch):
        d = _data()
        calls = []
        orig = pp.EyeData.copy
        monkeypatch.setattr(pp.EyeData, "copy",
                            lambda self, *a, **k: calls.append(1) or orig(self, *a, **k))
        d | (PREPROC | pp.crop(100, 2900))
        assert len(calls) == 1
        d | PREPROC | pp.crop(100, 2900)  # (d | PREPROC) | crop: two copies
        assert len(calls) == 3

    def test_object_inplace_setting_ignored(self):
        """steps run in place on the copy, also with an object-level inplace=False"""
        d = _data()
        d.inplace = False
        r = d | (pp.pupil.blinks_detect(strategies=["zero"]) | pp.crop(100, 2900))
        assert {k: len(iv) for k, iv in r.get_blinks().items()}["left_pupil"] == 1
        assert {k: len(iv) for k, iv in d.get_blinks().items()}["left_pupil"] == 0

    def test_same_as_method_chain(self):
        d = _data()
        r = d | PREPROC
        m = d.copy()
        m.pupil.blinks_detect(strategies=["zero"]).pupil.blinks_merge_close(distance=20)
        m.pupil.blinks_interpolate().pupil.lowpass_filter(5)
        np.testing.assert_array_equal(r.data["left_pupil"], m.data["left_pupil"])
        np.testing.assert_array_equal(r.data.mask["left_pupil"], m.data.mask["left_pupil"])

    def test_reuse_across_subjects(self):
        data = {s: _data(s) for s in ("s1", "s2")}
        out = {s: d | PREPROC for s, d in data.items()}
        assert [d.name for d in out.values()] == ["s1", "s2"]
        assert all({k: len(iv) for k, iv in d.get_blinks().items()}["left_pupil"] == 1 for d in out.values())
        assert all({k: len(iv) for k, iv in d.get_blinks().items()}["left_pupil"] == 0 for d in data.values())

    def test_history_per_step(self):
        r = _data() | PREPROC | pp.downsample(100)
        names = [h["funcstring"] for h in r.history[-5:]]
        assert names == ["pupil.blinks_detect(strategies=['zero'])",
                         "pupil.blinks_merge_close(distance=20)",
                         "pupil.blinks_interpolate()", "pupil.lowpass_filter(5)",
                         "downsample(100)"]

    def test_generic_and_setter_steps(self):
        d = _data()
        ev = d.events.keep(d.events.labels == "a")
        r = (d | pp.set_events(ev) | pp.gaze.mask_offscreen()
             | pp.exclude_intervals([(0, 100)]) | pp.reset_time())
        assert len(r.events) == 2 and len(d.events) == 3
        assert "offscreen" in r.data.layer_names
        assert "offscreen" not in d.data.layer_names

    def test_pipeline_can_be_pickled(self):
        """e.g. to send it to worker processes, one per subject"""
        import pickle
        p = pickle.loads(pickle.dumps(PREPROC | pp.downsample(100)))
        assert repr(p) == repr(PREPROC | pp.downsample(100))
        d = _data()
        np.testing.assert_array_equal((d | p).data["left_pupil"],
                                      (d | PREPROC | pp.downsample(100)).data["left_pupil"])

    def test_step_call_is_pipe(self):
        d = _data()
        s = pp.crop(100, 2000)
        assert len(s(d)) == len(d | s) and len(d) == 3000

    def test_inplace_update_operator(self):
        d = _data()
        orig = d
        d |= pp.crop(100, 2000)
        assert d is not orig and len(orig) == 3000


class TestErrors:
    def test_failing_step_named(self):
        d = _data().drop_data(variables="pupil", inplace=False)
        with pytest.raises(ValueError, match="no pupil data") as e:
            d | (pp.crop(0, 2000) | pp.pupil.blinks_detect())
        assert e.value.__notes__ == ["in step 2 of 2 of the pipeline: pp.pupil.blinks_detect()"]
        with pytest.raises(ValueError) as e:
            d | pp.crop(0, 2000) | pp.pupil.blinks_detect()
        assert e.value.__notes__ == ["in the pipeline step pp.pupil.blinks_detect()"]

    def test_pipe_into_non_data(self):
        with pytest.raises(TypeError, match="type int into pp.crop"):
            3 | pp.crop(0, 100)
        with pytest.raises(TypeError, match="type list into"):
            [_data()] | PREPROC
        with pytest.raises(TypeError, match="only EyeData"):
            PREPROC(np.zeros(10))

    def test_pipe_non_step(self):
        d = _data()
        with pytest.raises(TypeError, match="unsupported operand"):
            d | 3
        with pytest.raises(TypeError, match="unsupported operand"):
            pp.crop(0, 100) | 3
        with pytest.raises(TypeError, match="a Pipeline holds steps"):
            Pipeline([pp.crop(0, 100), 3])

    def test_missing_call(self):
        d = _data()
        with pytest.raises(TypeError, match=r"pp.pupil.blinks_detect is not a step yet"):
            d | pp.pupil.blinks_detect
        with pytest.raises(TypeError, match=r"pp.crop is not a step yet"):
            d | pp.crop
        with pytest.raises(TypeError, match=r"pp.crop is not a step yet"):
            pp.pupil.blinks_detect() | pp.crop


def test_namespace_objects():
    assert isinstance(pp.pupil, Namespace) and pp.pupil._data is None
    assert repr(pp.gaze) == "<pp.gaze: mask_divergences, mask_offscreen>"
