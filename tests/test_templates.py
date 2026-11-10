"""Shared helpers of the HTML displays (pypillometry.templates)."""
import base64

import numpy as np

from pypillometry.templates import _bin_means, render, sparkline


def test_bin_means_ignores_nan():
    values = np.array([1., np.nan, 3., np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(_bin_means(values, 3), [1., 3., np.nan])


def test_sparkline_is_png_data_uri():
    uri = sparkline(np.sin(np.linspace(0, 10, 50)), x=np.linspace(-1, 1, 50), mark_x=0)
    prefix = "data:image/png;base64,"
    assert uri.startswith(prefix)
    assert base64.b64decode(uri[len(prefix):])[:8] == b"\x89PNG\r\n\x1a\n"


def test_sparkline_bins_long_and_masked_data():
    values = np.ma.masked_array(np.arange(100_000.), mask=np.arange(100_000) % 3 == 0)
    assert sparkline(values).startswith("data:image/png")
    assert sparkline(np.full(10, np.nan)) is not None   # nothing to draw, still an image


def test_sparkline_empty():
    assert sparkline(np.array([])) is None


def test_render_escapes():
    html = render("segmented_repr.html", class_name="X", name="<b>", overview=[],
                  channels=[{"label": "a&b", "image": None}], columns="segments",
                  epochs=None)
    assert "&lt;b&gt;" in html and "a&amp;b" in html and "<img" not in html
