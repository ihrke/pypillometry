"""Tests for reading GazePoint recordings (Open Gaze API columns).

The recordings are written by the tests: 60 Hz, 20 samples, a blink at
samples 5-8 (LPV = 0, LPUPILV = 1 and the last pupil size repeated, as in
real recordings) and two dropped samples (12, 13).
"""
import numpy as np
import pytest

import pypillometry as pp

COLS = ["CNT", "TIME", "LPOGX", "LPOGY", "LPOGV", "RPOGX", "RPOGY", "RPOGV",
        "LPD", "LPV", "RPD", "RPV", "LEYEZ", "LPUPILD", "LPUPILV",
        "REYEZ", "RPUPILD", "RPUPILV", "USER"]


def _recording(n=20, drop=(12, 13)):
    rows = []
    for i in range(n):
        if i in drop:
            continue
        blink = 5 <= i <= 8
        lpd = 20. + 0.1 * i
        rows.append(dict(
            CNT=i, TIME=10. + i / 60.,
            LPOGX=0.5, LPOGY=0.25, LPOGV=0 if blink else 1,
            RPOGX=0.5, RPOGY=0.25, RPOGV=0 if blink else 1,
            LPD=20.4 if blink else lpd, LPV=0 if blink else 1,
            RPD=21., RPV=1,
            LEYEZ=0.6, LPUPILD=0.00488 if blink else lpd * 0.00024, LPUPILV=1,
            REYEZ=0.6, RPUPILD=0.005, RPUPILV=0 if i == 3 else 1,
            USER="VAR OLD" if i < 2 else ("VAR TRIAL 1" if i < 10 else "VAR TRIAL 2")))
    return rows


def _write(path, rows, sep="\t"):
    lines = [sep.join(COLS)] + [sep.join(str(r[c]) for c in COLS) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture
def fname(tmp_path):
    return _write(tmp_path / "rec_1.tsv", _recording())


class TestReader:
    def test_tables(self, fname):
        res = pp.read_gazepoint(fname)
        assert len(res["samples"]) == 18 and "TIME" not in res["samples"].columns
        assert res["samples"]["time"].iloc[0] == pytest.approx(10000.)
        # the label of the first sample is left over: not an event
        assert list(res["events"]["label"]) == ["VAR TRIAL 1", "VAR TRIAL 2"]
        assert res["events"]["time"].iloc[0] == pytest.approx(10000. + 2000. / 60.)
        assert res["info"]["duplicate_times"] == 0

    def test_initial_label(self, fname):
        res = pp.read_gazepoint(fname, keep_initial_label=True)
        assert list(res["events"]["label"]) == ["VAR OLD", "VAR TRIAL 1", "VAR TRIAL 2"]

    def test_comma_separated(self, tmp_path):
        res = pp.read_gazepoint(_write(tmp_path / "rec.csv", _recording(), sep=","))
        assert len(res["samples"]) == 18

    def test_unsorted_and_duplicate_times(self, tmp_path):
        rows = _recording()
        rows = rows[10:] + rows[:10] + [dict(rows[3], CNT=99)]
        res = pp.read_gazepoint(_write(tmp_path / "rec.tsv", rows))
        assert np.all(np.diff(res["samples"]["time"]) > 0)
        assert len(res["samples"]) == 18 and res["info"]["duplicate_times"] == 1

    def test_not_a_recording(self, tmp_path):
        fname = tmp_path / "other.tsv"
        fname.write_text("a\tb\n1\t2\n")
        with pytest.raises(ValueError, match="not a GazePoint recording"):
            pp.read_gazepoint(str(fname))


class TestFromGazepoint:
    def test_pupil_mm(self, fname):
        d = pp.EyeData.from_gazepoint(fname)
        assert d.name == "rec_1" and len(d) == 20  # the dropped samples are filled in
        assert d.fs == pytest.approx(60.)
        left = d.data["left_pupil"]
        # the blink is masked although LPUPILV is 1 (LPV is 0)
        assert np.all(np.isnan(left[5:9])) and np.all(np.isnan(left[12:14]))
        assert left[0] == pytest.approx(4.8)
        assert np.sum(~np.isnan(left)) == 14
        # RPUPILV == 0 is masked
        assert np.isnan(d.data["right_pupil"][3])
        assert d.info["pupil_units"] == "mm"
        assert d.info["eye_distance_mm"] == pytest.approx(600.)

    def test_pupil_px(self, fname):
        d = pp.EyeData.from_gazepoint(fname, pupil="px")
        left = d.data["left_pupil"]
        assert left[0] == pytest.approx(20.) and np.all(np.isnan(left[5:9]))
        # RPUPILV does not matter for the size in pixels
        assert d.data["right_pupil"][3] == pytest.approx(21.)
        with pytest.raises(ValueError, match="'mm' or 'px'"):
            pp.EyeData.from_gazepoint(fname, pupil="cm")

    def test_gaze(self, fname):
        d = pp.EyeData.from_gazepoint(fname)
        assert d.data["left_x"][0] == pytest.approx(0.5) and np.isnan(d.data["left_x"][6])
        assert d.experimental_setup is None
        d = pp.EyeData.from_gazepoint(fname, screen_resolution=(1920, 1080))
        assert d.data["left_x"][0] == pytest.approx(960.)
        assert d.data["left_y"][0] == pytest.approx(270.)
        assert d.experimental_setup.screen_resolution == (1920, 1080)
        d = pp.EyeData.from_gazepoint(fname, gaze=False)
        assert sorted(d.data) == ["left_pupil", "right_pupil"]

    def test_events(self, fname):
        d = pp.EyeData.from_gazepoint(fname, label_prefix="VAR ")
        assert list(d.events.labels) == ["TRIAL 1", "TRIAL 2"]
        assert d.events.onsets[1] == pytest.approx(10000. + 10000. / 60.)

    def test_save_and_load(self, fname, tmp_path):
        d = pp.EyeData.from_gazepoint(fname)
        d.write_file(str(tmp_path / "rec.pkl"))
        res = pp.EyeData.from_file(str(tmp_path / "rec.pkl"))
        assert np.allclose(res.data["left_pupil"], d.data["left_pupil"], equal_nan=True)
