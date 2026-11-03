"""Intervals as plain spans: validation and row-wise metadata."""
import numpy as np
import pytest

import pypillometry as pp


class TestValidation:

    @pytest.mark.parametrize("pairs, msg", [
        ([(10, 5)], "start is after its end"),
        ([(0, 1), (np.nan, 5)], "must be finite"),
        ([(0, np.inf)], "must be finite"),
        ([(0, 1, 2)], "must be \\(start, end\\) pairs"),
        ([(None, 1)], "must be numbers"),
    ])
    def test_invalid_pairs(self, pairs, msg):
        with pytest.raises(ValueError, match=msg):
            pp.Intervals(pairs, units="ms")

    def test_error_names_the_interval(self):
        with pytest.raises(ValueError, match=r"interval 1 is \(30, 20\)"):
            pp.Intervals([(0, 10), (30, 20)], units="ms")

    def test_empty_and_zero_length_are_valid(self):
        assert len(pp.Intervals([], units="ms")) == 0
        assert pp.Intervals([(5, 5)], units="ms").intervals == [(5, 5)]

    def test_pad_does_not_invert(self):
        # finding 5.7: pad(-40, -40) gave (340, 310)
        b = pp.Intervals([(100, 200), (300, 350)], units="ms")
        with pytest.raises(ValueError, match=r"interval 1 \(300, 350\) end before it starts"):
            b.pad(-40, -40)
        assert b.pad(-20, -10).intervals == [(120, 190), (320, 340)]
        # shrinking to zero length is fine
        assert b.pad(-25, -25).intervals[1] == (325, 325)


class TestInfiniteBoundsInLists:
    """-inf/inf in plain lists stand for the start/end of the recording."""

    def make(self):
        return pp.PupilData(left_pupil=np.arange(100.), sampling_rate=100,
                            time=np.arange(100) * 10. + 1000)

    def test_remove_intervals(self):
        d = self.make()
        r = d.remove_intervals([(-np.inf, 1200), (1800, np.inf)], inplace=False)
        np.testing.assert_array_equal(r.tx, d.crop(1200, 1790, inplace=False).tx)

    def test_exclude_and_keep(self):
        d = self.make()
        e = d.exclude_intervals([(1500, np.inf)], inplace=False)
        assert e.get_excluded().intervals == [(1500., 2000.)]
        k = d.keep_intervals([(-np.inf, 1500)], inplace=False)
        assert k.get_excluded().intervals == [(1500., 2000.)]


class TestNoEventLocking:
    """Intervals have no event-locking metadata (that is Epochs)."""

    @pytest.mark.parametrize("kwarg", ["interval_window", "event_indices"])
    def test_removed_arguments(self, kwarg):
        with pytest.raises(TypeError, match=kwarg):
            pp.Intervals([(0, 1)], units="ms", **{kwarg: (0, 1)})

    def test_format2_state_keys(self):
        st = pp.Intervals([(0, 1)], units="ms")._to_state()
        assert st["interval_window"] is None and st["event_indices"] is None
        # old states with these keys load
        iv = pp.Intervals._from_state({**st, "interval_window": (-5, 5),
                                       "event_indices": [3]})
        assert iv.intervals == [(0, 1)] and not hasattr(iv, "interval_window")


class TestNoDataRange:
    """Intervals do not store the range of the data; nothing is clipped to it."""

    def test_removed_argument(self):
        with pytest.raises(TypeError, match="data_time_range"):
            pp.Intervals([(0, 1)], units="ms", data_time_range=(0, 10))

    def test_conversions_not_clipped(self):
        d = pp.PupilData(left_pupil=np.ones(100), sampling_rate=100)
        iv = pp.Intervals([(-50, 20), (980, 1100)], units="ms", event_onsets=[-50, 1050])
        ix = iv.to_units("samples", d)
        assert ix.intervals == [(-5, 2), (98, 110)]
        assert list(ix.event_onsets) == [-5, 105]
        # but masks only cover the data
        assert iv.to_mask(d).sum() == 4

    def test_complement_needs_the_data(self):
        d = pp.PupilData(left_pupil=np.ones(100), sampling_rate=100)
        assert pp.Intervals([(10, 20)], "samples").complement(d).intervals == \
            [(0, 10), (20, 100)]
