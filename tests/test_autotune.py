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

    def test_max_duration(self, transplants):
        # blinks longer than max_duration are not replaced: the whole artifact counts
        train, _ = transplants
        good = dict(smooth_winsize=11, vel_onset=-5, vel_offset=5, margin=(5, 5), blinkwindow=100)
        err = transplant_error_mahot(train, **good)
        assert transplant_error_mahot(train, max_duration=10_000, **good) == err
        short = transplant_error_mahot(train, max_duration=5, **good)
        assert short > 5 * err and np.isclose(short, transplant_error_mahot(
            train, smooth_winsize=11, vel_onset=-1e9, vel_offset=1e9, margin=(0, 0),
            blinkwindow=1, max_duration=0))

    def test_several_signals(self, transplants):
        train, test = transplants
        both = transplant_error_mahot([train, test], per_blink=True, **BAD)
        assert len(both) == len(train) + len(test)
        assert np.isnan(transplant_error_mahot([], **BAD))

    def test_other_missing_samples_are_interpolated_too(self):
        # a blink of the host right after a transplanted blink: its anchor
        # point falls into it, so they are merged (as in a recording)
        from dataclasses import replace
        from pypillometry.signal.autotune import _all_blinks
        y, clean, cores = make_signal(n=6000, nblinks=1, seed=3)
        tr = transplant_blinks(clean, (150, 250), pad=50, source=y, cores=cores, n=1, seed=1)
        a, b = tr.blinks[0]
        sig = tr.signal.copy()
        sig[b + 120:b + 160] = np.nan           # the host's own blink
        tr = replace(tr, signal=sig)
        np.testing.assert_array_equal(_all_blinks(tr), [[a, b], [b + 120, b + 160]])
        pars = dict(smooth_winsize=11, vel_onset=-5, vel_offset=5, margin=(5, 100), blinkwindow=100)
        _, _, skipped = interpolate_blinks_mahot(sig, tr.blinks, **pars)
        assert skipped                          # alone, the blink is skipped
        err = transplant_error_mahot(tr, **pars)
        assert err < 0.2 * tr.error(sig)        # with the host's blink, it is interpolated


class TestTune:
    def test_improves_bad_parameters(self, transplants):
        train, test = transplants
        res = tune_blinks_mahot(train, test, start=BAD, seed=1, **FAST)
        assert res.success and res.params == res.tuned and res.start == BAD
        assert res.loss < 0.5 * res.loss_start
        assert res.test_loss < 0.5 * res.test_loss_start
        assert (res.n_blinks, res.n_test) == (len(train), len(test))
        assert np.isclose(res.loss, transplant_error_mahot(train, **res.params))
        assert res.loss >= res.history["loss"].min()   # changes below `tol` are not made
        assert set(res.history) == {"smooth_winsize", "vel_onset", "vel_offset", "margin_before",
                                    "margin_after", "blinkwindow", "loss", "stage", "search"}
        assert (res.history["vel_onset"] < 0).all() and res.params["vel_onset"] < 0
        assert res.params["smooth_winsize"] % 2 == 1
        filled, interp, _ = interpolate_blinks_mahot(train.signal, train.blinks, **res.params)
        assert not np.isnan(filled[train.sites[0, 0]:train.sites[0, 1]]).any()
        assert "tuned" in repr(res)

    def test_several_searches(self, transplants):
        train, test = transplants
        half = len(train.blinks) // 2
        sets = [transplant_blinks(train.clean, (150, 250), pad=50, source=train.signal,
                                  cores=train.blinks[:half], seed=1),
                transplant_blinks(train.clean, (150, 250), pad=50, source=train.signal,
                                  cores=train.blinks[half:], seed=2)]
        res = tune_blinks_mahot([[sets[0]], [sets[1]]], refine=test, start=BAD, seed=1, **FAST)
        assert res.n_searches == 2 and res.n_blinks == len(sets[0]) + len(sets[1])
        assert set(res.history["search"]) == {-1, 0, 1}
        assert (res.history["stage"][res.history["search"] >= 0] == 1).all()
        assert res.loss < 0.5 * res.loss_start
        assert "2 searches" in repr(res)

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

    def test_refine(self, transplants):
        train, test = transplants
        res = tune_blinks_mahot(train, refine=test, start=BAD, seed=1, **FAST)
        assert (res.n_blinks, res.n_refine, res.n_test) == (len(train), len(test), 0)
        assert np.isclose(res.loss, transplant_error_mahot([train, test], **res.params))
        assert np.isclose(res.loss_start, transplant_error_mahot([train, test], **BAD))
        assert set(res.history["stage"]) == {1, 2} and res.loss < 0.5 * res.loss_start

    def test_crop_keeps_errors(self, transplants):
        from pypillometry.signal.autotune import _NAMES, _context, _crop, _default_bounds, _to_params
        y, clean, _ = make_signal()
        # a few blinks in a long clean signal
        tr = transplant_blinks(np.r_[clean, clean], (150, 250), pad=50, n=3, source=y, seed=1)
        ranges = _default_bounds([tr])
        lo, hi = (np.array([ranges[k][i] for k in _NAMES]) for i in (0, 1))
        cropped = _crop(tr, _context([tr], hi))
        assert len(cropped.signal) < len(tr.signal) and len(cropped) == len(tr)
        rng = np.random.default_rng(1)
        for _ in range(10):
            params = _to_params(lo + rng.random(len(_NAMES)) * (hi - lo))
            np.testing.assert_allclose(transplant_error_mahot(cropped, per_blink=True, **params),
                                       transplant_error_mahot(tr, per_blink=True, **params))

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
    """Binocular pupil, 60 s at 500 Hz, with detected blinks."""
    left, _, _ = make_signal(n=30000, nblinks=20, seed=1)
    right, _, _ = make_signal(n=30000, nblinks=20, seed=2)
    return pp.EyeData(left_pupil=left, right_pupil=right, sampling_rate=500, name="blinks",
                      inplace=False).pupil.blinks_detect()


def record_transplants(monkeypatch):
    """list that gets the transplants made by blinks_optimize_parameters()"""
    from pypillometry.signal import transplant
    made, orig = [], transplant.transplant_blinks
    monkeypatch.setattr(transplant, "transplant_blinks",
                        lambda *a, **kw: made.append(orig(*a, **kw)) or made[-1])
    return made


class TestBlinksOptimizeParameters:
    def test_tunes_in_ms(self, data):
        before = data.data["left_pupil"].copy()
        start = dict(vel_onset=-200, vel_offset=200, margin=(0, 0))
        res = data.pupil.blinks_optimize_parameters(start=start, seed=1, **FAST)
        np.testing.assert_array_equal(data.data["left_pupil"], before)
        assert res.units == "ms" and res.success
        assert set(res.params) == {"winsize", "vel_onset", "vel_offset", "margin", "blinkwindow"}
        assert res.start == dict(winsize=26.0, vel_onset=-200.0, vel_offset=200.0,
                                 margin=(0.0, 0.0), blinkwindow=500.0)
        assert res.test_loss < 0.5 * res.test_loss_start
        assert res.n_test > 0 and res.n_refine > 0
        assert "winsize" in res.history and "winsize" in res.bounds
        assert set(res.history["stage"]) == {1, 2}
        assert res.bounds["margin_after"] == (0.0, 250.0)   # up to the window

    def test_subsets_of_blinks(self, data):
        # 10 blinks for the search, each 3 times; the others once
        res = data.pupil.blinks_optimize_parameters(eyes="left", seed=1, **FAST)
        assert 10 < res.n_blinks <= 30 and res.n_refine + res.n_test < 15   # as many as fit
        # several searches, each on n_tune blinks
        res = data.pupil.blinks_optimize_parameters(eyes="left", n_starts=2, n_tune=4, repeats=1,
                                                    n_refine=2, n_test=2, seed=1, **FAST)
        assert res.n_searches == 2 and res.n_blinks <= 8
        # at most n_refine and n_test of the others
        res = data.pupil.blinks_optimize_parameters(n_refine=3, n_test=4, seed=1, **FAST)
        assert 0 < res.n_refine <= 3 and 0 < res.n_test <= 4
        # few blinks: no refinement, half of them for testing
        res = data.pupil.blinks_optimize_parameters(eyes="left", n_tune=100, repeats=1, seed=1, **FAST)
        assert res.n_refine == 0 and abs(res.n_blinks - res.n_test) <= 1
        assert set(res.history["stage"]) == {1}

    def test_excluded_and_masked_samples_are_avoided(self, data, monkeypatch):
        made = record_transplants(monkeypatch)
        d = data.exclude_intervals([(0, 20000)])                     # samples 0-10000
        d = d.mask_intervals(pp.Intervals([(40000, 60000)], "ms"), variables="pupil")
        d.pupil.blinks_optimize_parameters(eyes="left", seed=1, **FAST)
        tuning = [tr for tr in made if len(tr)]
        assert tuning
        for tr in tuning:
            for iv in (tr.donors, tr.windows):
                assert (iv[:, 0] - tr.window[0] >= 10000).all() and (iv[:, 1] + tr.window[1] <= 20000).all()

    def test_arguments(self, data):
        res = data.pupil.blinks_optimize_parameters(eyes="left", n_test=0, n_tune=5, repeats=2,
                                         seed=1, fixed=dict(blinkwindow=300, winsize=20),
                                         bounds=dict(margin_after=(0, 100)), **FAST)
        assert res.n_test == 0 and res.n_blinks == 10 and res.success
        assert res.params["blinkwindow"] == 300 and res.params["winsize"] == 22
        assert res.params["margin"][1] <= 100
        with pytest.raises(ValueError):
            data.pupil.blinks_optimize_parameters(method="kalman")
        with pytest.raises(ValueError):
            data.pupil.blinks_optimize_parameters(start=dict(vel_onset="5%"))
        with pytest.raises(ValueError):
            data.pupil.blinks_optimize_parameters(fixed=dict(interp_type="linear"))
        with pytest.raises(ValueError, match="detect the blinks"):
            pp.EyeData(left_pupil=data.data["left_pupil"], sampling_rate=500,
                       inplace=False).pupil.blinks_optimize_parameters()
        loss = data.data["left_pupil"].copy()
        loss[np.isnan(loss)] = 1000         # no blinks, only tracker loss
        for s in range(1000, 29000, 1000):
            loss[s:s + 60] = np.nan
        with pytest.raises(ValueError, match="obvious"):
            pp.EyeData(left_pupil=loss, sampling_rate=500, inplace=False) \
                .pupil.blinks_detect().pupil.blinks_optimize_parameters()
        with pytest.raises(ValueError, match="obvious"):     # cores of 120 ms
            data.pupil.blinks_optimize_parameters(max_core=100)
        with pytest.raises(ValueError):
            data.pupil.blinks_optimize_parameters(n_tune=0)
        with pytest.raises(ValueError):
            data.pupil.blinks_optimize_parameters(n_test=-1)
        with pytest.raises(ValueError):    # interpolated
            data.pupil.blinks_detect().pupil.blinks_interpolate().pupil.blinks_optimize_parameters()


def test_only_obvious_blinks_are_transplanted(monkeypatch):
    # blinks and tracker loss (missing samples without transients); both
    # are detected, only the blinks are transplanted
    y, _, cores = make_signal(n=30000, nblinks=20, seed=1)
    loss = cores[::4, 0] + 300
    for s in loss:
        y[s:s + 60] = np.nan
    d = pp.EyeData(left_pupil=y, sampling_rate=500, inplace=False).pupil.blinks_detect()
    assert len(d.get_blinks("left", "pupil")["left_pupil"]) == len(cores) + len(loss)
    made = record_transplants(monkeypatch)
    d.pupil.blinks_optimize_parameters(seed=1, **FAST)
    donors = np.concatenate([tr.donors for tr in made if len(tr)])
    assert len(donors) > 0
    assert not np.isin(donors[:, 0], loss).any()
    assert np.isin(donors[:, 0], cores[:, 0]).all()
