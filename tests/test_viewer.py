import unittest

pytest = __import__("pytest")
pytest.importorskip("vispy")

import numpy as np

from pypillometry import Intervals
from pypillometry.viewer import _selections_to_intervals, _intervals_to_axis


class TestViewerSelections(unittest.TestCase):
    def test_seconds(self):
        res = _selections_to_intervals({"pupil": [(1.0, 2.0)]})
        self.assertEqual(res["pupil"].units, "sec")
        self.assertEqual(res["pupil"].intervals, [(1.0, 2.0)])

    def test_sample_index(self):
        """raw arrays without time: selections are sample indices"""
        res = _selections_to_intervals({"Signal": [(10, 20)]}, sample_index=True)
        self.assertIsNone(res["Signal"].units)
        self.assertEqual(res["Signal"].label, "selected_Signal")


class TestViewerHighlight(unittest.TestCase):
    def test_index_on_sample_axis(self):
        """arrays without time: index intervals are not converted"""
        iv = Intervals([(10, 20)], units=None)
        res = _intervals_to_axis(iv, np.arange(100), sample_index=True)
        self.assertEqual(res, [(10, 20)])

    def test_time_units_on_sample_axis(self):
        iv = Intervals([(10, 20)], units="ms", sampling_rate=1000)
        res = _intervals_to_axis(iv, np.arange(100), sample_index=True)
        self.assertEqual(res, [(10, 20)])

    def test_index_on_time_axis(self):
        """index intervals are placed using the time vector"""
        t = 5.0 + np.arange(100) * 0.01
        iv = Intervals([(10, 20), (90, 100)], units=None)
        res = _intervals_to_axis(iv, t)
        np.testing.assert_allclose(res, [(5.1, 5.2), (5.9, 6.0)])

    def test_ms_on_time_axis(self):
        iv = Intervals([(100, 200)], units="ms")
        self.assertEqual(_intervals_to_axis(iv, np.arange(10)), [(0.1, 0.2)])

    def test_intervals_not_modified(self):
        iv = Intervals([(10, 20)], units=None)
        _intervals_to_axis(iv, np.arange(100) * 0.01)
        self.assertIsNone(iv.sampling_rate)
        self.assertIsNone(iv.units)
        self.assertEqual(iv.intervals, [(10, 20)])


if __name__ == '__main__':
    unittest.main()
