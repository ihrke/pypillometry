"""Table, RecordDict and EyeData.layer_stats()."""
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


def test_record_dict():
    two = pp.RecordDict({"a": {"x": {"n": 1}, "y": {"n": 2}}, "b": {"x": {"n": 3}}},
                        index_names=("channel", "layer"))
    assert two["a"]["y"]["n"] == 2 and isinstance(two, dict)
    df = two.to_pandas()
    assert list(df.index) == [("a", "x"), ("a", "y"), ("b", "x")]
    assert list(df.index.names) == ["channel", "layer"] and list(df["n"]) == [1, 2, 3]
    one = pp.RecordDict({"a": {"n": 1, "m": "u"}}, index_names=("name",))
    assert one.to_pandas().index.name == "name" and one.to_pandas().loc["a", "m"] == "u"
    assert "<table" in two._repr_html_() and "channel" in repr(two)
    assert pp.RecordDict().to_pandas().empty


def test_dataset_dict():
    ds = pp.get_available_datasets()
    assert isinstance(ds, pp.RecordDict)
    assert "rlmw_002_short" in ds and "description" in ds["rlmw_002"]
    df = ds.to_pandas()
    assert df.loc["rlmw_010_edf", "filetype"] == "edf" and df.index.name == "dataset"


def data():
    """12 samples at 100 Hz: blinks at 2-4 and 8."""
    d = pp.EyeData(left_pupil=np.arange(12.), left_x=np.zeros(12), left_y=np.zeros(12),
                   sampling_rate=100)
    d.set_blinks(pp.Intervals([(2, 4), (8, 9)], "samples"), eyes="left", variables="pupil")
    return d


def test_layer_stats_entries_and_values():
    d = data()
    stats = d.layer_stats()
    assert isinstance(stats, pp.RecordDict)
    # the layers that are set on a channel, and "masked" for every channel
    assert {ch: list(layers) for ch, layers in stats.items()} == {
        "left_x": ["masked"], "left_y": ["masked"], "left_pupil": ["blink", "masked"]}
    b = stats["left_pupil"]["blink"]
    assert b["n"] == 2 and b["units"] == "ms"
    assert b["total_duration"] == 30. and b["max"] == 20. and b["min"] == 10.
    assert b["percent"] == pytest.approx(25.)              # 3 of 12 samples
    assert type(b["percent"]) is float
    assert stats["left_x"]["masked"]["n"] == 0
    assert d.layer_stats(units="samples")["left_pupil"]["blink"]["total_duration"] == 3
    df = stats.to_pandas()
    assert ("left_pupil", "blink") in df.index and "percent" in df.columns


def test_layer_stats_excluded_and_policy():
    d = data()
    d.exclude_intervals(pp.Intervals([(10, 12)], "samples"))
    stats = d.layer_stats(eyes="left", variables="pupil")
    assert list(stats["left_pupil"]) == ["blink", "masked"]
    # percentages and stretches within the included 10 samples
    assert stats["left_pupil"]["blink"]["percent"] == pytest.approx(30.)
    assert stats["left_pupil"]["masked"]["n"] == 2        # the excluded period is not a stretch
    with pytest.raises(ValueError, match="excluded periods apply to all channels"):
        d.layer_stats("excluded")
    # explicit layers, also layers that are not set
    stats = d.layer_stats(["offscreen", "masked"], eyes="left", variables="pupil", policy="raw")
    assert list(stats["left_pupil"]) == ["offscreen", "masked"]
    assert stats["left_pupil"]["offscreen"]["n"] == 0
    assert np.isnan(stats["left_pupil"]["offscreen"]["mean"])


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
    stats = d.layer_stats()
    for eye in ("left", "right"):
        assert q[f"artifacts.{eye}.interpolated_pct"] == pytest.approx(
            stats[f"{eye}_pupil"]["interpolated"]["percent"])
        assert q[f"artifacts.{eye}.offscreen_pct"] == pytest.approx(
            stats[f"{eye}_x"]["offscreen"]["percent"])
