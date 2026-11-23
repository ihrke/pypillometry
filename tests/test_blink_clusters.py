"""Groups of detected blinks (d.plot.plot_blink_clusters())."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pypillometry as pp
from pypillometry.plot.pupil import (_blink_segments, _centered_segments, _cluster_blinks,
                                     _has_missing)


def make_data(n=60000, seed=0):
    """Smooth signal with blinks (drop, 60 missing samples, recovery) and
    gaps without transients (tracker loss), every 600 samples."""
    rng = np.random.default_rng(seed)
    y = 1000 + np.convolve(rng.normal(0, 30, n + 400), np.hanning(401) / 200, "valid")
    y += rng.normal(0, 1, n)
    starts = np.arange(600, n - 600, 600)
    loss = starts[::4]
    for s in starts:
        if s not in loss:
            amp = rng.uniform(300, 500)
            y[s - 20:s] -= np.linspace(0, amp, 20)
            y[s + 60:s + 100] -= np.linspace(amp, 0, 40)
        y[s:s + 60] = np.nan
    return y, starts, loss


class TestHelpers:
    def test_segments(self):
        y = np.arange(100, dtype=float) + 1
        y[40:50] = np.nan
        y[47] = 5                       # a valid sample in the core
        y[80] = 0                       # missing value code
        blinks = np.array([[35, 55], [60, 70], [78, 83], [0, 3]])
        before, after, core = _blink_segments(y, blinks, 5, 5)
        np.testing.assert_array_equal(_has_missing(y, blinks), [True, False, True, False])
        np.testing.assert_array_equal(core, [10, 0, 1, 0])
        # core 40-50: before it 35-39, after it 50-54, each relative to its outer end
        np.testing.assert_allclose(before[0], np.arange(36, 41) - 36)
        np.testing.assert_allclose(after[0], np.arange(51, 56) - 55)
        # core 80 (missing value code); no missing samples: NaN
        np.testing.assert_allclose(after[2], np.arange(82, 87) - 86)
        assert np.isnan(before[[1, 3]]).all() and np.isnan(after[[1, 3]]).all()
        # around the middle, relative to the start; outside the data: NaN
        mid = _centered_segments(y, blinks[[1, 3]], 5, 5)
        np.testing.assert_allclose(mid[0], np.arange(61, 71) - 61)
        assert np.isnan(mid[1, :4]).all() and np.allclose(np.diff(mid[1, 4:]), 1)

    def test_clusters(self):
        y, starts, loss = make_data()
        blinks = np.c_[starts, starts + 60]
        before, after, core = _blink_segments(y, blinks, 100, 250)
        labels = _cluster_blinks(before, after, core, 2, 0.02, 2, np.random.default_rng(1))
        is_loss = np.isin(starts, loss)
        # blinks and tracker loss are separate groups, the blinks the larger one
        assert set(labels[~is_loss]) == {0} and set(labels[is_loss]) == {1}
        # a subset is clustered, the rest goes to the nearest group
        sub = _cluster_blinks(before, after, core, 2, 0.02, 2, np.random.default_rng(1),
                              max_n=40)
        np.testing.assert_array_equal(sub, labels)

    def test_small_groups_are_other(self):
        # two groups of blinks are needed; the three tracker losses (4%) split
        # off before, but are fewer than min_size
        y, starts, loss = make_data()
        keep = ~np.isin(starts, loss) | np.isin(starts, loss[:3])
        blinks = np.c_[starts[keep], starts[keep] + 60]
        before, after, core = _blink_segments(y, blinks, 100, 250)
        labels = _cluster_blinks(before, after, core, 2, 0.1, 2, np.random.default_rng(1))
        is_loss = np.isin(blinks[:, 0], loss)
        assert (labels[is_loss] == -1).all()
        assert set(labels[~is_loss]) == {0, 1}


class TestPlot:
    def teardown_method(self):
        plt.close("all")

    def test_panels(self):
        y, starts, loss = make_data()
        d = pp.EyeData(left_pupil=y, sampling_rate=500, inplace=False).pupil.blinks_detect()
        # a dip without missing samples, found by the velocity
        y[30300:30320] -= np.r_[np.linspace(0, 300, 10), np.linspace(300, 0, 10)]
        d = pp.EyeData(left_pupil=y, sampling_rate=500, inplace=False).pupil.blinks_detect()
        fig = d.plot.plot_blink_clusters(n_clusters=2, window=(200, 500))
        axs, hists = fig.axes[:4], fig.axes[4:]   # the panels, then the histograms below them
        assert len(axs) == 4 and len(hists) == 4
        assert all(h.get_xscale() == "log" for h in hists)
        assert hists[0].get_xlabel().startswith("core (closed) duration")
        assert hists[3].get_xlabel().startswith("blink duration")
        titles = [ax.get_title() for ax in axs]
        assert titles[0].startswith("left, group 1: ") and "other" in titles[2]
        assert titles[3].startswith("left, without missing samples: 1 ")
        # all blinks with missing samples in grey in each panel of the groups
        n = len(d.get_blinks("left", "pupil")["left_pupil"]) - 1
        grey = [l for l in axs[0].lines if l.get_color() == "0.85"]
        assert [np.isnan(l.get_ydata()).sum() for l in grey] == [n, n]
        assert "without missing samples" in axs[0].get_ylabel()

    def test_errors(self):
        y, _, _ = make_data()
        d = pp.EyeData(left_pupil=y, sampling_rate=500, inplace=False)
        with pytest.raises(ValueError, match="detect the blinks"):
            d.plot.plot_blink_clusters()
        d = d.pupil.blinks_detect()
        with pytest.raises(ValueError):
            d.plot.plot_blink_clusters(n_clusters=0)
        with pytest.raises(ValueError):
            d.plot.plot_blink_clusters(min_size=1)
        with pytest.raises(ValueError):
            d.plot.plot_blink_clusters(window=2)
        with pytest.raises(ValueError, match="5 axes needed"):
            d.plot.plot_blink_clusters(axes=plt.subplots(1, 3)[1])
