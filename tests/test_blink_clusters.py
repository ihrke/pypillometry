"""Groups of detected blinks (d.plot.plot_blink_clusters())."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pypillometry as pp
from pypillometry.plot.pupil import _blink_segments, _cluster_blinks


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
        np.testing.assert_array_equal(core, [10, 0, 1, 0])
        # core 40-50: before it 35-39, after it 50-54, each relative to its outer end
        np.testing.assert_allclose(before[0], np.arange(36, 41) - 36)
        np.testing.assert_allclose(after[0], np.arange(51, 56) - 55)
        # no missing samples: the smallest sample (60)
        np.testing.assert_allclose(after[1], np.arange(62, 67) - 66)
        # outside the data: NaN
        assert np.isnan(before[3]).all()

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
        fig = d.plot.plot_blink_clusters(n_clusters=2, window=(200, 500))
        axs = fig.axes
        assert len(axs) == 3
        titles = [ax.get_title() for ax in axs]
        assert titles[0].startswith("left, group 1: ") and "other" in titles[2]
        # all blinks in grey in each panel
        n = len(d.get_blinks("left", "pupil")["left_pupil"])
        grey = [l for l in axs[0].lines if l.get_color() == "0.85"]
        assert sum(np.isnan(l.get_ydata()).sum() >= n for l in grey) == 2

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
        with pytest.raises(ValueError, match="4 axes needed"):
            d.plot.plot_blink_clusters(axes=plt.subplots(1, 3)[1])
