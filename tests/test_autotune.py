"""Tuning the interpolation of blinks on transplanted blinks."""
import numpy as np
import pytest

import pypillometry as pp
from pypillometry.signal import (interpolate_blinks_mahot, transplant_blinks,
                                 transplant_error_mahot, tune_blinks_mahot)
from tests.test_transplant import make_signal

# far too short margins and too high thresholds: the artifacts stay in
BAD = dict(smooth_winsize=11, vel_onset=-400, vel_offset=400, margin=(0, 0), blinkwindow=100)
FAST = dict(n_grid=5, sweeps=2)


@pytest.fixture(scope="module")
def transplants():
    y, clean, cores = make_signal()
    return (transplant_blinks(y, (150, 250), pad=50, cores=cores[:25], seed=1),
            transplant_blinks(y, (150, 250), pad=50, cores=cores[25:], seed=2))


class TestError:
    def test_mean_error_per_blink(self, transplants):
        train, test = transplants
        err = transplant_error_mahot(train, **BAD)
        per_blink = transplant_error_mahot(train, per_blink=True, **BAD)
        assert per_blink.shape == (len(train),) and np.isclose(per_blink.mean(), err)
        good = transplant_error_mahot(train, 11, -5, 5, (5, 5), 100)
        assert good < 0.5 * err

    def test_several_signals(self, transplants):
        train, test = transplants
        both = transplant_error_mahot([train, test], per_blink=True, **BAD)
        assert len(both) == len(train) + len(test)
        assert np.isnan(transplant_error_mahot([], **BAD))


class TestTune:
    def test_improves_bad_parameters(self, transplants):
        train, test = transplants
        res = tune_blinks_mahot(train, test, start=BAD, seed=1, **FAST)
        assert res.success and res.params == res.tuned and res.start == BAD
        assert res.loss < 0.5 * res.loss_start
        assert res.test_loss < 0.5 * res.test_loss_start
        assert (res.n_blinks, res.n_test) == (len(train), len(test))
        assert np.isclose(res.loss, transplant_error_mahot(train, **res.params))
        assert np.isclose(res.loss, res.history["loss"].min())
        assert set(res.history) == {"smooth_winsize", "vel_onset", "vel_offset", "margin_before",
                                    "margin_after", "blinkwindow", "loss"}
        assert (res.history["vel_onset"] < 0).all() and res.params["vel_onset"] < 0
        assert res.params["smooth_winsize"] % 2 == 1
        filled, interp, _ = interpolate_blinks_mahot(train.signal, train.blinks, **res.params)
        assert not np.isnan(filled[train.sites[0, 0]:train.sites[0, 1]]).any()
        assert "tuned" in repr(res)

    def test_reproducible(self, transplants):
        a = tune_blinks_mahot(transplants[0], start=BAD, seed=3, **FAST)
        b = tune_blinks_mahot(transplants[0], start=BAD, seed=3, **FAST)
        assert a.params == b.params and a.loss == b.loss
        assert np.isnan(a.test_loss) and a.success

    def test_bounds_and_fixed(self, transplants):
        res = tune_blinks_mahot(transplants[0], start=BAD, seed=1,
                                bounds={"margin_before": (10, 20), "vel_onset": (-50, -1)},
                                fixed={"blinkwindow": 77, "smooth_winsize": 9}, **FAST)
        assert res.params["blinkwindow"] == 77 and res.params["smooth_winsize"] == 9
        assert (res.history["blinkwindow"][1:] == 77).all()
        assert 10 <= res.params["margin"][0] <= 20
        assert -50 <= res.params["vel_onset"] <= -1
        assert res.bounds["vel_onset"] == (-50, -1) and res.bounds["blinkwindow"] == (77, 77)
        res = tune_blinks_mahot(transplants[0], start=BAD, fixed={"margin": (3, 4)}, seed=1, **FAST)
        assert res.params["margin"] == (3, 4)

    def test_without_start(self, transplants):
        res = tune_blinks_mahot(transplants[0], seed=1, **FAST)
        assert res.start is None and np.isnan(res.loss_start) and res.success
        with pytest.raises(ValueError):
            tune_blinks_mahot(transplants[0], n_init=0)

    def test_from_start_only(self, transplants):
        res = tune_blinks_mahot(transplants[0], start=BAD, n_init=0, seed=1)
        assert res.loss < 0.5 * res.loss_start

    def test_keeps_start_if_not_better_on_test(self, transplants):
        train, test = transplants
        everything = dict(BAD)
        res = tune_blinks_mahot(train, test, start=BAD, fixed=everything, **FAST)
        assert not res.success and res.params == BAD

    def test_errors(self, transplants):
        y = np.ones(1000)
        with pytest.raises(ValueError):
            tune_blinks_mahot(transplant_blinks(y, 50), start=BAD)
        with pytest.raises(ValueError):
            tune_blinks_mahot(transplants[0], bounds={"margin": (0, 1)})
        with pytest.raises(ValueError):
            tune_blinks_mahot(transplants[0], fixed={"winsize": 5})


@pytest.fixture(scope="module")
def data():
    """Binocular pupil, 60 s at 500 Hz, with blinks."""
    left, _, _ = make_signal(n=30000, nblinks=20, seed=1)
    right, _, _ = make_signal(n=30000, nblinks=20, seed=2)
    return pp.EyeData(left_pupil=left, right_pupil=right, sampling_rate=500, name="blinks",
                      inplace=False)


class TestBlinksTransplant:
    def test_transplanted_object(self, data):
        d = data.pupil.blinks_detect()
        dt = d.pupil.blinks_transplant(window=(300, 500), pad=100, seed=1)
        assert dt is not d and "left_pupilorig" not in d.data
        np.testing.assert_array_equal(dt.data["left_pupilorig"], d.data["left_pupil"])
        sites = dt.get_layer_intervals("transplant", variables="pupil", units="samples")
        assert set(sites) == {"left_pupil", "right_pupil"} and len(sites["left_pupil"]) > 3
        # the cores are blinks, the windows are not masked
        nblinks = lambda obj: len(obj.get_blinks("left", "pupil")["left_pupil"])
        assert nblinks(dt) == nblinks(d) + len(sites["left_pupil"])
        a, b = sites["left_pupil"].intervals[0]
        masked = dt.mask("left", "pupil")[a:b]
        assert np.isnan(dt.data["left_pupil"][a:b]).sum() == 60 == masked.sum()
        assert not np.isnan(d.data["left_pupil"][a:b]).any()
        assert b - a == 50 + 150 + 60 + 250 + 50
        assert "blinks_transplant" in dt.history[-1]["funcstring"]

    def test_arguments(self, data):
        dt = data.pupil.blinks_transplant(eyes="left", n=3, original_as=None, seed=1)
        assert "left_pupilorig" not in dt.data
        assert len(dt.get_layer_intervals("transplant", "left", "pupil")["left_pupil"]) == 3
        assert not dt.data.get_layer("right_pupil", "transplant").any()
        same = data.pupil.blinks_transplant(eyes="left", n=3, seed=1)
        np.testing.assert_array_equal(same.data["left_pupil"], dt.data["left_pupil"])
        # nothing can be transplanted: no blinks of this size
        none = data.pupil.blinks_transplant(min_duration=500, max_duration=None)
        assert not none.data.get_layer("left_pupil", "transplant").any()

    def test_error(self, data):
        dt = data.pupil.blinks_detect().pupil.blinks_transplant(seed=1)
        raw = dt.pupil.blinks_transplant_error()
        di = dt.pupil.blinks_interpolate(vel_onset=-2, vel_offset=2)
        err = di.pupil.blinks_transplant_error()
        assert set(err) == {"left_pupil", "right_pupil"}
        assert err["left_pupil"] < 0.3 * raw["left_pupil"]
        per_blink = di.pupil.blinks_transplant_error("left", per_blink=True)["left_pupil"]
        assert len(per_blink) == len(di.get_layer_intervals("transplant", "left", "pupil")["left_pupil"])
        with pytest.raises(ValueError):
            data.pupil.blinks_transplant_error()
        with pytest.raises(ValueError):
            di.pupil.blinks_transplant()     # interpolated

    def test_masked_samples_are_avoided(self, data):
        d = data.mask_intervals(pp.Intervals([(0, 30000)], "ms"), eyes="left", variables="pupil")
        dt = d.pupil.blinks_transplant(seed=1)
        sites = dt.get_layer_intervals("transplant", "left", "pupil")["left_pupil"]
        assert len(sites) > 0 and min(a for a, b in sites.intervals) >= 30000

    def test_step(self, data):
        dt = data | pp.pupil.blinks_transplant(n=2, seed=1)
        assert dt.data.get_layer("left_pupil", "transplant").any()


class TestBlinksAutotune:
    def test_tunes_in_ms(self, data):
        before = data.data["left_pupil"].copy()
        start = dict(vel_onset=-200, vel_offset=200, margin=(0, 0))
        res = data.pupil.blinks_autotune(start=start, seed=1, **FAST)
        np.testing.assert_array_equal(data.data["left_pupil"], before)
        assert res.units == "ms" and res.success
        assert set(res.params) == {"winsize", "vel_onset", "vel_offset", "margin", "blinkwindow"}
        assert res.start == dict(winsize=26.0, vel_onset=-200.0, vel_offset=200.0,
                                 margin=(0.0, 0.0), blinkwindow=500.0)
        assert res.test_loss < 0.5 * res.test_loss_start
        assert res.n_test > 0 and "winsize" in res.history and "winsize" in res.bounds
        assert res.bounds["margin_after"] == (0.0, 500.0)
        # the parameters give this error
        d = data.pupil.blinks_detect()
        dt = d.pupil.blinks_transplant(seed=5)
        tuned = dt.pupil.blinks_interpolate(**res.params, inplace=False).pupil.blinks_transplant_error()
        bad = dt.pupil.blinks_interpolate(**res.start, inplace=False).pupil.blinks_transplant_error()
        assert tuned["left_pupil"] < 0.5 * bad["left_pupil"]

    def test_arguments(self, data):
        res = data.pupil.blinks_autotune(eyes="left", test_fraction=0, n=10, seed=1,
                                         fixed=dict(blinkwindow=300, winsize=20),
                                         bounds=dict(margin_after=(0, 100)), **FAST)
        assert res.n_test == 0 and res.n_blinks == 10 and res.success
        assert res.params["blinkwindow"] == 300 and res.params["winsize"] == 22
        assert res.params["margin"][1] <= 100
        with pytest.raises(ValueError):
            data.pupil.blinks_autotune(method="kalman")
        with pytest.raises(ValueError):
            data.pupil.blinks_autotune(start=dict(vel_onset="5%"))
        with pytest.raises(ValueError):
            data.pupil.blinks_autotune(fixed=dict(interp_type="linear"))
        with pytest.raises(ValueError):
            data.pupil.blinks_autotune(min_duration=500)   # no blinks of this size
