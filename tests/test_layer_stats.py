"""Table and EyeData.layer_stats()."""
import numpy as np
import pytest

import pypillometry as pp


def test_table():
    t = pp.Table({"channel": ["a", "b"]}, n=[1, 2])
    assert isinstance(t, dict) and t.nrows == 2
    assert isinstance(t["n"], np.ndarray)
    assert list(t.to_pandas().columns) == ["channel", "n"]
    assert "channel" in repr(t) and "<table" in t._repr_html_()
    assert pp.Table().nrows == 0
    with pytest.raises(ValueError, match="same length.*a: 2, b: 1"):
        pp.Table(a=[1, 2], b=[1])


def data():
    """12 samples at 100 Hz: blinks at 2-4 and 8, excluded 10-12."""
    d = pp.EyeData(left_pupil=np.arange(12.), left_x=np.zeros(12), left_y=np.zeros(12),
                   sampling_rate=100)
    d.set_blinks(pp.Intervals([(2, 4), (8, 9)], "samples"), eyes="left", variables="pupil")
    return d


def rows(t):
    return {(c, l): i for i, (c, l) in enumerate(zip(t["channel"].tolist(), t["layer"].tolist()))}


def test_layer_stats_rows_and_values():
    d = data()
    t = d.layer_stats()
    assert isinstance(t, pp.Table)
    # layers that are set on a channel, and "masked" for every channel
    assert list(rows(t)) == [("left_x", "masked"), ("left_y", "masked"),
                             ("left_pupil", "blink"), ("left_pupil", "masked")]
    i = rows(t)[("left_pupil", "blink")]
    assert t["n"][i] == 2
    assert t["total_duration"][i] == 30. and t["max"][i] == 20. and t["min"][i] == 10.
    assert t["percent"][i] == pytest.approx(25.)          # 3 of 12 samples
    assert t["n"][rows(t)[("left_x", "masked")]] == 0
    assert d.layer_stats(units="samples")["total_duration"][i] == 3


def test_layer_stats_excluded_and_policy():
    d = data()
    d.exclude_intervals(pp.Intervals([(10, 12)], "samples"))
    t = d.layer_stats(eyes="left", variables="pupil")
    r = rows(t)
    assert list(r) == [("left_pupil", "blink"), ("left_pupil", "masked"), ("all", "excluded")]
    # percentages and stretches within the included 10 samples
    assert t["percent"][r[("left_pupil", "blink")]] == pytest.approx(30.)
    assert t["n"][r[("left_pupil", "masked")]] == 2       # the excluded period is not a stretch
    assert t["percent"][r[("all", "excluded")]] == pytest.approx(100 * 2 / 12)
    # explicit layers, also layers that are not set
    t = d.layer_stats(["offscreen", "masked"], eyes="left", variables="pupil", policy="raw")
    assert list(rows(t)) == [("left_pupil", "offscreen"), ("left_pupil", "masked")]
    assert t["n"].tolist() == [0, 0] and np.isnan(t["mean"][0])


def test_blink_stats_is_a_shortcut():
    d = data()
    stats = d.blink_stats()
    assert stats == {"left_pupil": {"n": 2, "mean": 15., "sd": 5., "min": 10., "max": 20.,
                                    "total_duration": 30., "units": "ms"}}
    assert all(type(k) is str for k in stats)
    assert not hasattr(d, "nblinks")
    assert d.summary()["nblinks"] == {"left_pupil": 2}


def test_quality_report_agrees():
    """the percentages of the quality report are those of layer_stats"""
    d = pp.get_example_data("rlmw_002_short")
    d = d | pp.pupil.blinks_detect() | pp.pupil.blinks_interpolate() | pp.gaze.mask_offscreen()
    d.exclude_intervals([(0, 3000)])
    q = d.quality_report().to_dict(flat=True)
    t = d.layer_stats()
    r = rows(t)
    for eye in ("left", "right"):
        assert q[f"artifacts.{eye}.interpolated_pct"] == pytest.approx(
            t["percent"][r[(f"{eye}_pupil", "interpolated")]])
        assert q[f"artifacts.{eye}.offscreen_pct"] == pytest.approx(
            t["percent"][r[(f"{eye}_x", "offscreen")]])
