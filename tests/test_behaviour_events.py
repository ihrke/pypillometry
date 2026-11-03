"""
Tests for events from behaviour logs: Events.from_table(), sync_clocks(),
add_events() and event_onset().
"""
import unittest
import numpy as np
import pandas as pd
import pypillometry as pp
from pypillometry.events import Events


def fake_data():
    """Short recording (0-10 s) with two events."""
    tx = np.arange(0, 10000, 2.0)
    return pp.PupilData(time=tx, left_pupil=np.ones_like(tx),
                        event_onsets=[1000, 5000],
                        event_labels=["Start Block 1", "Start Block 2"],
                        sampling_rate=500, inplace=False)


class TestFromTable(unittest.TestCase):

    def setUp(self):
        self.behav = pd.DataFrame({"stim": [1.0, 3.0, 5.0],
                                   "rt": [0.5, np.nan, 0.25]})

    def test_columns_list(self):
        ev = Events.from_table(self.behav, ["stim"], units="sec")
        np.testing.assert_array_equal(ev.onsets, [1, 3, 5])
        np.testing.assert_array_equal(ev.labels, ["stim"] * 3)
        self.assertEqual(ev.units, "sec")

    def test_dict_with_arrays_drops_nan_and_sorts(self):
        ev = Events.from_table(self.behav, {"stim": "stim",
                               "response": self.behav.stim + self.behav.rt},
                               units="sec")
        np.testing.assert_array_equal(ev.onsets, [1, 1.5, 3, 5, 5.25])
        np.testing.assert_array_equal(
            ev.labels, ["stim", "response", "stim", "stim", "response"])

    def test_equal_onsets_keep_column_order(self):
        ev = Events.from_table({"a": [1.0], "b": [1.0]}, ["b", "a"], units="ms")
        np.testing.assert_array_equal(ev.labels, ["b", "a"])

    def test_default_units_ms(self):
        self.assertEqual(Events.from_table(self.behav, ["stim"]).units, "ms")

    def test_different_lengths_raise(self):
        with self.assertRaises(ValueError):
            Events.from_table(self.behav, {"stim": "stim", "x": [1.0]})


class TestSyncClocks(unittest.TestCase):

    def setUp(self):
        self.src = np.array([10.0, 70.0, 130.0, 400.0])           # sec
        self.dst = 5000 + self.src * 1000 * (1 + 2e-5)           # ms, drifting

    def test_fit_offset_and_drift(self):
        sync = pp.sync_clocks(self.src, self.dst, src_units="sec",
                              dst_units="ms")
        self.assertAlmostEqual(sync.offset, 5000)
        self.assertAlmostEqual(sync.drift, 2e-5)
        np.testing.assert_allclose(sync.residuals, 0, atol=1e-6)
        self.assertAlmostEqual(sync(200.0), 5000 + 200000 * (1 + 2e-5))

    def test_offset_only(self):
        sync = pp.sync_clocks(self.src, self.dst, src_units="sec",
                              dst_units="ms", slope=False)
        self.assertEqual(sync.drift, 0)
        self.assertEqual(sync.slope, 1000)
        self.assertGreater(np.abs(sync.residuals).max(), 1)

    def test_single_pair_gives_offset(self):
        sync = pp.sync_clocks([2.0], [7000.0], src_units="sec", dst_units="ms")
        self.assertEqual(sync.slope, 1000)
        self.assertEqual(sync.offset, 5000)

    def test_units_from_quantities_and_events(self):
        dst = Events(self.dst, ["sync"] * 4, units="ms")
        sync = pp.sync_clocks(self.src * pp.u.sec, dst)
        self.assertEqual((sync.src_units, sync.dst_units), ("sec", "ms"))
        self.assertAlmostEqual(sync(1 * pp.u.min), sync(60.0))

    def test_map_events(self):
        sync = pp.sync_clocks(self.src, self.dst, src_units="sec",
                              dst_units="ms")
        ev = sync(Events([10000.0], ["x"], units="ms"))   # 10 s on src clock
        self.assertEqual(ev.units, "ms")
        np.testing.assert_allclose(ev.onsets, self.dst[:1])
        np.testing.assert_array_equal(ev.labels, ["x"])

    def test_wrong_pairing_warns(self):
        with _capture_warnings() as msgs:
            pp.sync_clocks(self.src, self.dst[[0, 2, 1, 3]], src_units="sec",
                           dst_units="ms")
        self.assertTrue(any("paired correctly" in m for m in msgs))

    def test_errors(self):
        with self.assertRaises(ValueError):       # lengths
            pp.sync_clocks(self.src, self.dst[:3], src_units="sec",
                           dst_units="ms")
        with self.assertRaises(ValueError):       # missing units
            pp.sync_clocks(self.src, self.dst, dst_units="ms")
        with self.assertRaises(ValueError):       # NaN
            pp.sync_clocks([1.0, np.nan], [1.0, 2.0], src_units="sec",
                           dst_units="ms")
        with self.assertRaises(ValueError):       # empty
            pp.sync_clocks([], [], src_units="sec", dst_units="ms")


class _capture_warnings:
    """Collect loguru warnings."""
    def __enter__(self):
        self.msgs = []
        self.id = pp.logger.add(lambda m: self.msgs.append(str(m)),
                                level="WARNING")
        return self.msgs

    def __exit__(self, *args):
        pp.logger.remove(self.id)


class TestAddEvents(unittest.TestCase):

    def test_merge_and_sort(self):
        d = fake_data()
        d2 = d.add_events(Events([3.0, 0.5], ["stim", "stim"], units="sec"))
        np.testing.assert_array_equal(d2.events.onsets, [500, 1000, 3000, 5000])
        np.testing.assert_array_equal(
            d2.events.labels, ["stim", "Start Block 1", "stim", "Start Block 2"])
        self.assertEqual(len(d.events), 2)      # copy, not in-place

    def test_inplace(self):
        d = fake_data()
        self.assertIs(d.add_events(Events([2000], ["x"]), inplace=True), d)
        self.assertEqual(len(d.events), 3)

    def test_existing_first_at_same_time(self):
        d = fake_data().add_events(Events([1000], ["new"]))
        np.testing.assert_array_equal(d.events.labels[:2], ["Start Block 1", "new"])

    def test_errors(self):
        d = fake_data()
        with self.assertRaises(TypeError):
            d.add_events([1000])
        with self.assertRaises(ValueError):
            d.add_events(Events([20000], ["late"]))

    def test_history(self):
        d = fake_data().add_events(Events([2000], ["x"]))
        self.assertIn("add_events", str(d.history))


class TestEventOnset(unittest.TestCase):

    def test_onset(self):
        d = fake_data()
        self.assertEqual(d.event_onset("Start Block 2"), 5000)
        self.assertEqual(d.event_onset("Start Block 2", units="sec"), 5)

    def test_exact_match_only(self):
        with self.assertRaises(ValueError):
            fake_data().event_onset("Start Block")    # substring of two

    def test_not_unique(self):
        d = fake_data().add_events(Events([2000], ["Start Block 1"]))
        with self.assertRaises(ValueError):
            d.event_onset("Start Block 1")


if __name__ == "__main__":
    unittest.main()
