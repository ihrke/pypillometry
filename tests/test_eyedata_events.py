"""
The events stored in eyedata objects (one Events object, see d.events).
"""
import os
import tempfile
import unittest

import numpy as np

import pypillometry as pp
from pypillometry import state


def make_data():
    """10 s at 100 Hz from 1000 ms, five events with metadata."""
    tx = 1000 + np.arange(1000) * 10.
    d = pp.EyeData(left_pupil=np.full(1000, 5.), time=tx, sampling_rate=100,
                     event_onsets=[1500, 3000, 5000, 7000, 9000],
                     event_labels=["a", "b", "c", "d", "e"])
    return d.set_events(d.events.set_meta("k", [1, 2, 3, 4, 5]))


class TestStoredEvents(unittest.TestCase):

    def test_events_is_the_stored_object(self):
        d = make_data()
        self.assertIs(d.events, d.events)
        self.assertIsInstance(d.events, pp.Events)
        self.assertEqual(d.events.units, "ms")

    def test_old_attributes_removed(self):
        d = make_data()
        for name in ("event_onsets", "event_labels", "event_meta",
                     "get_events", "set_event_onsets"):
            self.assertFalse(hasattr(d, name), name)

    def test_events_cannot_be_changed_in_place(self):
        d = make_data()
        with self.assertRaises(TypeError):
            d.events.meta["x"] = np.zeros(5)
        with self.assertRaises(ValueError):
            d.events.meta["k"][0] = 10
        with self.assertRaises(ValueError):
            d.events.onsets[0] = 2000
        with self.assertRaises(AttributeError):
            d.events = d.events
        np.testing.assert_array_equal(d.events.meta["k"], [1, 2, 3, 4, 5])
        self.assertEqual(list(d.events.meta), ["k"])

    def test_constructor_without_labels(self):
        d = pp.EyeData(left_pupil=np.ones(10), sampling_rate=10, event_onsets=[100, 200])
        np.testing.assert_array_equal(d.events.labels, ["", ""])
        d = pp.EyeData(left_pupil=np.ones(10), sampling_rate=10)
        self.assertEqual(len(d.events), 0)

    def test_constructor_validates(self):
        with self.assertRaises(ValueError):
            pp.EyeData(left_pupil=np.ones(10), sampling_rate=10,
                         event_onsets=[100, 200], event_labels=["a"])
        with self.assertRaises(ValueError):
            pp.EyeData(left_pupil=np.ones(10), sampling_rate=10,
                         event_onsets=[5000], event_labels=["a"])


class TestSetEvents(unittest.TestCase):

    def test_replaces_events(self):
        d = make_data()
        ev = d.events.filter("b")
        d2 = d.copy().set_events(ev)
        np.testing.assert_array_equal(d2.events.labels, ["b"])
        np.testing.assert_array_equal(d2.events.meta["k"], [2])
        self.assertEqual(len(d.events), 5)

    def test_converts_to_ms(self):
        d = make_data()
        d.set_events(pp.Events([2, 3], ["x", "y"], units="sec"))
        self.assertEqual(d.events.units, "ms")
        np.testing.assert_allclose(d.events.onsets, [2000, 3000])

    def test_not_events_raises(self):
        d = make_data()
        with self.assertRaises(TypeError):
            d.set_events(([1500], ["a"]))
        with self.assertRaises(TypeError):
            d.add_events({"onsets": [1500]})

    def test_onsets_outside_data_raise(self):
        d = make_data()
        for onsets in ([500], [20000]):
            with self.assertRaises(ValueError):
                d.set_events(pp.Events(onsets, ["x"]))
            with self.assertRaises(ValueError):
                d.add_events(pp.Events(onsets, ["x"]))
        self.assertEqual(len(d.events), 5)


class TestEventsFollowTime(unittest.TestCase):

    def test_reset_time(self):
        d = make_data().reset_time(inplace=False)
        np.testing.assert_array_equal(d.events.onsets, [500, 2000, 4000, 6000, 8000])
        np.testing.assert_array_equal(d.events.meta["k"], [1, 2, 3, 4, 5])
        d = make_data().reset_time(2000, inplace=False)
        np.testing.assert_array_equal(d.events.onsets, [-500, 1000, 3000, 5000, 7000])

    def test_crop(self):
        d = make_data().crop(3000, 7000, inplace=False)
        np.testing.assert_array_equal(d.events.onsets, [3000, 5000, 7000])
        np.testing.assert_array_equal(d.events.labels, ["b", "c", "d"])
        np.testing.assert_array_equal(d.events.meta["k"], [2, 3, 4])

    def test_remove_intervals(self):
        # removing 4000-6000 ms: "c" is deleted, later events move back 2000 ms
        d = make_data().remove_intervals([(4000, 6000)], inplace=False)
        np.testing.assert_array_equal(d.events.onsets, [1500, 3000, 5000, 7000])
        np.testing.assert_array_equal(d.events.labels, ["a", "b", "d", "e"])
        np.testing.assert_array_equal(d.events.meta["k"], [1, 2, 4, 5])

    def test_remove_intervals_at_the_ends(self):
        d = make_data().remove_intervals([(0, 2000), (8000, 20000)], inplace=False)
        np.testing.assert_array_equal(d.events.onsets, [3000, 5000, 7000])
        np.testing.assert_array_equal(d.events.meta["k"], [2, 3, 4])

    def test_inplace_false_keeps_original(self):
        d = make_data()
        d.crop(3000, 7000, inplace=False)
        d.remove_intervals([(4000, 6000)], inplace=False)
        d.reset_time(inplace=False)
        np.testing.assert_array_equal(d.events.onsets, [1500, 3000, 5000, 7000, 9000])


class TestEventsSaved(unittest.TestCase):

    def test_pickle_and_hdf5(self):
        d = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            for fname in ("d.pypil", "d.h5"):
                path = os.path.join(tmp, fname)
                d.write_file(path)
                d2 = pp.read_file(path)
                np.testing.assert_array_equal(d2.events.onsets, d.events.onsets)
                np.testing.assert_array_equal(d2.events.labels, d.events.labels)
                np.testing.assert_array_equal(d2.events.meta["k"], d.events.meta["k"])

    def test_state_has_events_state(self):
        st = state.to_state(make_data())
        self.assertNotIn("event_onsets", st)
        ev = st["events"]
        np.testing.assert_array_equal(ev["onsets"], [1500, 3000, 5000, 7000, 9000])
        np.testing.assert_array_equal(ev["labels"], ["a", "b", "c", "d", "e"])
        self.assertEqual(ev["units"], "ms")
        self.assertIsInstance(ev["meta"], dict)
        np.testing.assert_array_equal(ev["meta"]["k"], [1, 2, 3, 4, 5])

    def test_copy(self):
        d = make_data()
        c = d.copy()
        np.testing.assert_array_equal(c.events.meta["k"], d.events.meta["k"])


if __name__ == "__main__":
    unittest.main()
