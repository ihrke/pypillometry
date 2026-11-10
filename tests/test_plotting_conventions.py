"""
The plotting conventions of the package (see pypillometry/plot/_axes.py):

- single-panel plots draw on `ax` (no new figure) and return it
- multi-panel plots draw on the given `axes` or make their own figure and
  return the figure
- plots with several figures return the list of figures
"""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pypillometry as pp
from pypillometry import u
from pypillometry.plot._axes import get_ax, get_axes


class TestHelpers(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_get_ax(self):
        fig, ax = plt.subplots()
        self.assertIs(get_ax(ax), ax)
        self.assertIs(get_ax(), plt.gca())

    def test_get_axes_new_figure(self):
        fig, axes = get_axes(None, 3, figsize=(4, 5))
        self.assertEqual(len(axes), 3)
        self.assertEqual(len(fig.axes), 3)
        np.testing.assert_allclose(fig.get_size_inches(), (4, 5))
        fig, axes = get_axes(None, 2, nrows=1, ncols=2)
        self.assertEqual(len(axes), 2)

    def test_get_axes_given(self):
        fig, axs = plt.subplots(2, 2)
        f, axes = get_axes(axs, 4)
        self.assertIs(f, fig)
        self.assertEqual(axes, list(axs.ravel()))
        fig, ax = plt.subplots()
        self.assertEqual(get_axes(ax, 1)[1], [ax])
        with self.assertRaisesRegex(ValueError, "2 axes needed .one per variable"):
            get_axes(ax, 2, "variable")


class ConventionTest(unittest.TestCase):
    """Helpers to check the conventions"""

    @classmethod
    def setUpClass(cls):
        cls.d = pp.get_example_data("rlmw_002_short").pupil.blinks_detect()

    def tearDown(self):
        plt.close("all")

    def assertSinglePanel(self, plot, **kwargs):
        """`plot(ax=ax)` draws on ax, opens no figure and returns ax;
        without ax, it draws on the current axes"""
        fig, ax = plt.subplots()
        nfigs = len(plt.get_fignums())
        res = plot(ax=ax, **kwargs)
        self.assertIs(res, ax)
        self.assertEqual(len(plt.get_fignums()), nfigs)
        self.assertTrue(ax.has_data() or ax.texts or ax.patches)
        fig, ax = plt.subplots()
        self.assertIs(plot(**kwargs), ax)

    def assertMultiPanel(self, plot, n, **kwargs):
        """without axes: a new figure (returned); with axes: drawn there"""
        plt.figure()  # an existing figure is not reused
        nfigs = len(plt.get_fignums())
        fig = plot(**kwargs)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(plt.get_fignums()), nfigs + 1)
        fig2, axs = plt.subplots(1, n)
        nfigs = len(plt.get_fignums())
        res = plot(axes=axs, **kwargs)
        self.assertIs(res, fig2)
        self.assertEqual(len(plt.get_fignums()), nfigs)
        with self.assertRaisesRegex(ValueError, "axes needed"):
            plot(axes=[fig2.axes[0]] * (n + 1), **kwargs)


class TestSinglePanel(ConventionTest):
    def test_plot_pupil(self):
        self.assertSinglePanel(self.d.plot.plot_pupil, plot_range=(0, 5) * u.sec)

    def test_plot_pupil_title_kwargs(self):
        fig, ax = plt.subplots()
        self.d.plot.plot_pupil(plot_range=(0, 5) * u.sec, eyes="left", ax=ax,
                               title="T", color="red", show_events=False)
        self.assertEqual(ax.get_title(), "T")
        self.assertEqual(ax.lines[0].get_color(), "red")

    def test_plot_masked(self):
        self.assertSinglePanel(self.d.plot.plot_masked, variables="pupil")
        fig, ax = plt.subplots()
        self.d.plot.plot_masked(variables="pupil", ax=ax, title="T")
        self.assertEqual(ax.get_title(), "T")

    def test_events_plot(self):
        self.assertSinglePanel(self.d.events.filter("S_").plot, title="T")
        self.assertEqual(plt.gca().get_title(), "T")

    def test_intervals_plot(self):
        intervals = self.d.get_epochs("S_", (0, 1000) * u.ms).to_intervals("sec")
        self.assertSinglePanel(intervals.plot)
        fig, ax = plt.subplots()
        self.assertIs(pp.Intervals([], units="ms").plot(ax=ax), ax)

    def test_plot_prf(self):
        self.assertSinglePanel(pp.signal.plot_prf, title="T")

    def test_roi_plot(self):
        for roi in [pp.CircularROI((100, 100), radius=50),
                    pp.RectangularROI((0, 0), (10, 10))]:
            self.assertSinglePanel(roi.plot)

    def test_calibration_plot(self):
        from pypillometry.eyedata.spatial_calibration import SpatialCalibration
        points = np.array([[640, 512], [640, 87], [640, 936], [96, 512], [1183, 512]])
        errors = np.array([[0.3, -13, -2], [0.3, 13, -0.2], [0.42, 16, 9],
                           [0.79, -19, 30], [0.62, 13, 24]])
        cal = SpatialCalibration("left", "HV5", points, points + errors[:, 1:],
                                 errors, (1280, 1024))
        self.assertSinglePanel(cal.plot, title="T")
        self.assertEqual(plt.gca().get_title(), "T")

    def test_plot_scanpath(self):
        self.assertSinglePanel(self.d.plot.plot_scanpath, plot_range=(10, 12) * u.sec,
                               eyes="left", title="T")
        self.assertEqual(plt.gca().get_title(), "T")

    def test_segment_plots(self):
        seg = self.d.get_segments(self.d.get_epochs("S_", (-500, 1500) * u.ms), "left_pupil")
        self.assertSinglePanel(seg.plot)
        self.assertSinglePanel(seg.pupil.plot)
        self.assertSinglePanel(seg.plot_carpet)


class TestPlotRange(ConventionTest):
    def test_plot_range_is_first(self):
        """plots with a plot_range take it as the first argument"""
        import inspect
        for name in dir(self.d.plot):
            fct = getattr(self.d.plot, name)
            if name.startswith("_") or not callable(fct):
                continue
            params = list(inspect.signature(fct).parameters)
            if "plot_range" in params:
                with self.subTest(name):
                    self.assertEqual(params[0], "plot_range")

    def test_positional_plot_range(self):
        fig, ax = plt.subplots()
        self.d.plot.plot_pupil((0, 5) * u.sec, ax=ax, units="sec")
        self.assertLessEqual(ax.lines[0].get_xdata().max(), 5)
        fig = self.d.plot.plot_pupil_foreshortening_error_surface((0, 5) * u.sec, eyes="left")
        self.assertIsInstance(fig, plt.Figure)


class TestMultiPanel(ConventionTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.setup = pp.ExperimentalSetup(
            screen_resolution=(1920, 1080), physical_screen_size=("52 cm", "29 cm"),
            eye_to_screen_center="60 cm", camera_spherical=("20 deg", "-90 deg", "70 cm"))

    def test_setup_plot_2d(self):
        self.assertMultiPanel(self.setup.plot, 3)

    def test_setup_plot_3d(self):
        fig = self.setup.plot(projection="3d", title="T")
        self.assertEqual(fig.axes[0].get_title(), "T")
        fig2 = plt.figure()
        ax = fig2.add_subplot(projection="3d")
        self.assertIs(self.setup.plot(projection="3d", axes=ax), fig2)

    def test_plot_timeseries(self):
        self.assertMultiPanel(self.d.plot.plot_timeseries, 2, variables=["pupil", "x"],
                              plot_range=(0, 5) * u.sec)
        fig = self.d.plot.plot_timeseries(variables="pupil", eyes="left", title="T",
                                          color="red", figsize=(4, 3))
        self.assertEqual(fig._suptitle.get_text(), "T")
        self.assertEqual(fig.axes[0].lines[0].get_color(), "red")
        np.testing.assert_allclose(fig.get_size_inches(), (4, 3))

    def test_plot_heatmap(self):
        self.assertMultiPanel(self.d.plot.plot_heatmap, 2, eyes=["left", "right"])
        fig = self.d.plot.plot_heatmap(eyes="left", title="T")
        self.assertEqual(fig._suptitle.get_text(), "T")

    def test_plot_foreshortening_surface(self):
        self.assertMultiPanel(self.d.plot.plot_pupil_foreshortening_error_surface, 2,
                              eyes=["left", "right"])

    def test_plot_calibration(self):
        from pypillometry.eyedata.spatial_calibration import SpatialCalibration
        points = np.array([[640, 512], [640, 87], [640, 936], [96, 512], [1183, 512]])
        errors = np.array([[0.3, -13, -2], [0.3, 13, -0.2], [0.42, 16, 9],
                           [0.79, -19, 30], [0.62, 13, 24]])
        d = self.d.copy()
        d.calibration = {eye: SpatialCalibration(eye, "HV5", points, points + errors[:, 1:],
                                                 errors, (1280, 1024))
                         for eye in ["left", "right"]}
        self.assertMultiPanel(d.plot.plot_calibration, 2)
        fig = d.plot.plot_calibration(eyes="left", title="T", figsize=(5, 4))
        self.assertEqual(fig._suptitle.get_text(), "T")
        np.testing.assert_allclose(fig.get_size_inches(), (5, 4))


class TestMultiFigure(ConventionTest):
    def test_lists_of_figures(self):
        import tempfile, os
        epochs = self.d.get_epochs("S_", (-500, 1000) * u.ms)
        plots = {
            "plot_intervals": lambda **kw: self.d.plot.plot_intervals(epochs, **kw),
            "plot_blinks": lambda **kw: self.d.plot.plot_blinks(eyes="left", **kw),
            "plot_timeseries_segments": lambda **kw: self.d.plot.plot_timeseries_segments(
                interv=20 * u.sec, **kw),
            "plot_pupil_segments": lambda **kw: self.d.plot.plot_pupil_segments(
                interv=20 * u.sec, **kw),
        }
        for name, plot in plots.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                figs = plot(figsize=(4, 3))
                self.assertGreater(len(figs), 0)
                for fig in figs:
                    self.assertIsInstance(fig, plt.Figure)
                    np.testing.assert_allclose(fig.get_size_inches(), (4, 3))
                pdf = os.path.join(tmp, "sub", "plots.pdf")
                figs_pdf = plot(pdf_file=pdf)
                self.assertEqual(len(figs_pdf), len(figs))
                self.assertTrue(os.path.exists(pdf))
                # the figures written to the PDF are closed
                self.assertFalse(set(f.number for f in figs_pdf) & set(plt.get_fignums()))


if __name__ == "__main__":
    unittest.main()
