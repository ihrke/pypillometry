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


if __name__ == "__main__":
    unittest.main()
