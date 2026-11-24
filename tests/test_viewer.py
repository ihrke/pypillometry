import unittest

pytest = __import__("pytest")
pytest.importorskip("vispy")

import numpy as np

from pypillometry import Intervals
from pypillometry.viewer import _selections_to_intervals, _intervals_to_axis, _highlight_by_panel


class TestViewerSelections(unittest.TestCase):
    def test_seconds(self):
        res = _selections_to_intervals({"pupil": [(1.0, 2.0)]})
        self.assertEqual(res["pupil"].units, "sec")
        self.assertEqual(res["pupil"].intervals, [(1.0, 2.0)])

    def test_sample_index(self):
        """raw arrays without time: selections are sample indices"""
        res = _selections_to_intervals({"Signal": [(10, 20)]}, sample_index=True)
        self.assertEqual(res["Signal"].units, "samples")
        self.assertEqual(res["Signal"].label, "selected_Signal")


class TestViewerHighlight(unittest.TestCase):
    def test_index_on_sample_axis(self):
        """arrays without time: index intervals are not converted"""
        iv = Intervals([(10, 20)], units="samples")
        res = _intervals_to_axis(iv, np.arange(100), sample_index=True)
        self.assertEqual(res, [(10, 20)])

    def test_time_units_on_sample_axis(self):
        """without a time vector, time units cannot be placed"""
        iv = Intervals([(10, 20)], units="ms")
        with self.assertRaises(ValueError):
            _intervals_to_axis(iv, np.arange(100), sample_index=True)

    def test_index_on_time_axis(self):
        """index intervals are placed using the time vector"""
        t = 5.0 + np.arange(100) * 0.01
        iv = Intervals([(10, 20), (90, 100)], units="samples")
        res = _intervals_to_axis(iv, t)
        np.testing.assert_allclose(res, [(5.1, 5.2), (5.9, 6.0)])

    def test_ms_on_time_axis(self):
        iv = Intervals([(100, 200)], units="ms")
        self.assertEqual(_intervals_to_axis(iv, np.arange(10)), [(0.1, 0.2)])

    def test_intervals_not_modified(self):
        iv = Intervals([(10, 20)], units="samples")
        _intervals_to_axis(iv, np.arange(100) * 0.01)
        self.assertEqual(iv.units, "samples")
        self.assertEqual(iv.intervals, [(10, 20)])


class TestHighlightByPanel(unittest.TestCase):
    PANELS = ["pupil", "x", "extra_vel"]

    def test_intervals_to_all_panels(self):
        iv = Intervals([(1, 2)], units="sec")
        res = _highlight_by_panel(iv, self.PANELS)
        self.assertEqual(set(res), set(self.PANELS))
        self.assertEqual(_highlight_by_panel(None, self.PANELS), {})

    def test_panel_keys(self):
        iv = Intervals([(1, 2)], units="sec")
        res = _highlight_by_panel({"pupil": iv, "extra_vel": iv}, self.PANELS)
        self.assertEqual(set(res), {"pupil", "extra_vel"})

    def test_eye_keys_are_combined(self):
        """keys from get_blinks() go to the plot of the variable"""
        left = Intervals([(100, 200)], units="ms")
        right = Intervals([(500, 600)], units="ms")
        res = _highlight_by_panel({"left_pupil": left, "right_pupil": right,
                                   "left_x": Intervals([], units="ms"),
                                   "left_y": left}, self.PANELS)    # y not shown: fine
        self.assertEqual(set(res), {"pupil", "x"})
        self.assertEqual(res["pupil"].intervals, [(100, 200), (500, 600)])
        self.assertEqual(left.intervals, [(100, 200)])

    def test_unknown_keys_warn(self):
        from loguru import logger
        messages = []
        sink = logger.add(messages.append, level="WARNING")
        try:
            res = _highlight_by_panel({"blinks": Intervals([(1, 2)], units="sec")},
                                      self.PANELS)
        finally:
            logger.remove(sink)
        self.assertEqual(res, {})
        self.assertTrue(any("'blinks'" in m for m in messages))
        with self.assertRaises(TypeError):
            _highlight_by_panel([(1, 2)], self.PANELS)


class TestTweakKeys(unittest.TestCase):
    def press(self, key, text):
        from types import SimpleNamespace
        from unittest import mock
        from vispy.util import keys
        from pypillometry.viewer.tweak_canvas import TweakCanvas
        canvas = mock.Mock()
        canvas.navigation.handle_key_press.return_value = None
        event = SimpleNamespace(key=keys.Key(key), text=text, modifiers=[])
        TweakCanvas.on_key_press(canvas, event)
        return canvas._show_help.called

    def test_help_keys(self):
        self.assertTrue(self.press('H', 'h'))
        self.assertTrue(self.press('?', '?'))
        self.assertFalse(self.press('/', '/'))


if __name__ == '__main__':
    unittest.main()


class TestAxisWidget(unittest.TestCase):
    def test_domain_follows_layout(self):
        """axis ticks match the view after the layout moves view and axis
        (wrong time axis in pp.tweak(), bug 1.10)"""
        from vispy import scene
        from vispy.geometry import Rect
        from pypillometry.viewer.visuals import _AxisWidget

        root = scene.Widget()
        vb = scene.ViewBox(parent=root)
        vb.camera = scene.PanZoomCamera(aspect=None)
        ax = _AxisWidget(orientation="bottom")
        ax.parent = root
        ax.link_view(vb)
        vb.rect = Rect(0, 0, 1000, 500)
        ax.size, ax.pos = (1000, 50), (0, 500)
        vb.camera.set_range(x=(0, 10), y=(0, 1))
        cam = vb.camera.rect
        # the layout moves the view and the axis (grid sets the size of
        # the axis before its position), then the window is resized
        for x, w in [(100, 1000), (100, 500)]:
            vb.rect = Rect(x, 0, w, 500)
            ax.size, ax.pos = (w, 50), (x, 500)
            np.testing.assert_allclose(ax.axis.domain, (cam.left, cam.right), atol=1e-4)


class TestMinMaxDownsample(unittest.TestCase):
    def downsample(self, data, mask=None, factor=4):
        from pypillometry.viewer.visuals import _minmax_downsample
        data = np.asarray(data, dtype=np.float32)
        time = np.arange(len(data), dtype=np.float32)
        if mask is None:
            mask = np.zeros(len(data), dtype=bool)
        return _minmax_downsample(time, data, np.asarray(mask, dtype=bool), factor)

    def test_min_and_max_in_order(self):
        """each chunk gives its minimum and maximum, in order of occurrence;
        samples after the last full chunk are dropped"""
        normal, masked = self.downsample([1, 5, 0, 2,  3, -1, 7, 4,  9])
        np.testing.assert_array_equal(normal, [(1, 5), (2, 0), (5, -1), (6, 7)])
        self.assertTrue(np.all(np.isnan(masked[:, 1])))
        self.assertEqual(normal.dtype, np.float32)

    def test_non_finite(self):
        """non-finite samples are ignored, chunks without finite samples
        are NaN"""
        nan, inf = np.nan, np.inf
        normal, _ = self.downsample([nan, 2, inf, 1,  nan, inf, -inf, nan])
        np.testing.assert_array_equal(normal, [(1, 2), (3, 1), (4, nan), (7, nan)])

    def test_mask(self):
        """masked samples are in `masked`, the others in `normal`"""
        nan = np.nan
        normal, masked = self.downsample([1, 5, 0, 2], mask=[0, 1, 0, 0])
        np.testing.assert_array_equal(normal, [(1, nan), (2, 0)])
        np.testing.assert_array_equal(masked, [(1, 5), (2, nan)])

    def test_shorter_than_chunk(self):
        normal, _ = self.downsample([3, 1, 2], factor=10)
        np.testing.assert_array_equal(normal, [(0, 3), (1, 1)])


class TestShareFonts(unittest.TestCase):
    def test_one_font_manager(self):
        """text, tick labels and axis labels use the same font manager"""
        from types import SimpleNamespace
        from vispy import scene
        from pypillometry.viewer.visuals import _share_fonts

        root = scene.Widget()
        ax = scene.AxisWidget(orientation="left", axis_label="y")
        ax.parent = root
        text = scene.Text("label", parent=root)
        _share_fonts(SimpleNamespace(scene=root))
        fonts = {id(t._font) for t in (text, ax.axis._text, ax.axis._axis_label_vis)}
        self.assertEqual(len(fonts), 1)


class TestTweakResult(unittest.TestCase):
    """the result of func is checked before the viewer opens"""

    def test_none(self):
        import pypillometry as pp
        with self.assertRaisesRegex(TypeError, "return its result"):
            pp.tweak(np.zeros(10), lambda x, a=1: None, {"a": 1})

    def test_scalar(self):
        import pypillometry as pp
        with self.assertRaisesRegex(TypeError, "1-d array"):
            pp.tweak(np.zeros(10), lambda x, a=1: 3.0, {"a": 1})

    def test_func_fails(self):
        import pypillometry as pp
        with self.assertRaises(RuntimeError):
            pp.tweak(np.zeros(10), lambda x, a=1: 1 / 0, {"a": 1})


class TestOverlayEyeData(unittest.TestCase):
    """overlay=: other EyeData objects on the plots of the same variables"""

    @classmethod
    def setUpClass(cls):
        import pypillometry as pp
        cls.pp = pp
        cls.d = pp.get_example_data("rlmw_002_short").pupil.blinks_detect()
        cls.k = cls.d.pupil.blinks_interpolate(method="kalman", inplace=False)
        cls.k.name = "kalman"

    def test_lines_per_eye_and_variable(self):
        from pypillometry.viewer import _overlays_from_eyedata
        res = _overlays_from_eyedata(self.k, len(self.d.tx))
        self.assertEqual(set(res), {"pupil", "x", "y"})
        self.assertEqual(set(res["pupil"]), {"kalman left", "kalman right"})
        np.testing.assert_array_equal(res["pupil"]["kalman left"], self.k.data["left_pupil"])
        # only the given eyes; the same name twice is numbered
        res = _overlays_from_eyedata([self.k, self.k], len(self.d.tx), eyes="left")
        self.assertEqual(set(res["pupil"]), {"kalman left", "kalman (2) left"})

    def test_errors(self):
        from pypillometry.viewer import _overlays_from_eyedata
        with self.assertRaisesRegex(TypeError, "EyeData"):
            _overlays_from_eyedata(np.zeros(10), 10)
        with self.assertRaisesRegex(ValueError, "samples"):
            _overlays_from_eyedata(self.k.crop(0, 10, units="sec", inplace=False),
                                   len(self.d.tx))
        # checked before the viewer opens
        with self.assertRaisesRegex(TypeError, "overlay_pupil must be a dict"):
            self.pp.view(self.d, overlay_pupil=self.k)
        with self.assertRaisesRegex(TypeError, "only be used when viewing an EyeData"):
            self.pp.view(np.zeros(10), overlay=self.k)
