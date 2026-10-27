import unittest

pytest = __import__("pytest")
pytest.importorskip("vispy")

from pypillometry.viewer import _selections_to_intervals


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


if __name__ == '__main__':
    unittest.main()
