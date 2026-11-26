"""Tests for reading GazePoint recordings (Open Gaze API columns).

The recordings are written by the tests: 60 Hz, 20 samples, a blink at
samples 5-8 (LPV = 0, LPUPILV = 1 and the last pupil size repeated, as in
real recordings) and a lost sample (12, a gap in CNT). TIME is rounded to
1/64 s, as on some computers, so that it jitters around the true time.
"""
import numpy as np
import pytest

import pypillometry as pp

COLS = ["CNT", "TIME", "LPOGX", "LPOGY", "LPOGV", "RPOGX", "RPOGY", "RPOGV",
        "LPCX", "LPCY", "LPD", "LPV", "RPCX", "RPCY", "RPD", "RPV",
        "LEYEZ", "LPUPILD", "LPUPILV", "REYEZ", "RPUPILD", "RPUPILV", "USER"]


def _recording(n=20, drop=(12,)):
    rows = []
    for i in range(n):
        if i in drop:
            continue
        blink = 5 <= i <= 8
        lpd = 20. + 0.1 * i
        rows.append(dict(
            CNT=100 + i, TIME=np.round((10. + i / 60.) * 64) / 64,
            LPOGX=0.5, LPOGY=0.25, LPOGV=0 if blink else 1,
            RPOGX=0.5, RPOGY=0.25, RPOGV=0 if blink else 1,
            LPCX=0.3, LPCY=0.4, LPD=20.4 if blink else lpd, LPV=0 if blink else 1,
            RPCX=0.7, RPCY=0.4, RPD=21., RPV=1,
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
        tab = res["samples"]
        assert len(tab) == 19 and {"time", "TIME", "CNT"} <= set(tab.columns)
        # the label of the first sample is left over: not an event
        assert list(res["events"]["label"]) == ["VAR TRIAL 1", "VAR TRIAL 2"]
        assert res["info"]["time_base"] == "CNT" and res["info"]["lost_samples"] == 1

    def test_time_from_cnt(self, fname):
        """The time is a straight line in CNT, not the jittering TIME."""
        res = pp.read_gazepoint(fname)
        t = res["samples"]["time"].to_numpy()
        cnt = res["samples"]["CNT"].to_numpy()
        # (19 samples with TIME rounded to 15.6 ms: the fit is good to 1 %)
        np.testing.assert_allclose(np.diff(t) / np.diff(cnt), 1000. / 60., rtol=1e-2)
        assert t[0] == pytest.approx(10000., abs=8.)
        assert res["info"]["sampling_rate"] == pytest.approx(60., rel=1e-2)
        assert res["events"]["time"].iloc[0] == pytest.approx(t[2])

    def test_initial_label(self, fname):
        res = pp.read_gazepoint(fname, keep_initial_label=True)
        assert list(res["events"]["label"]) == ["VAR OLD", "VAR TRIAL 1", "VAR TRIAL 2"]

    def test_comma_separated(self, tmp_path):
        res = pp.read_gazepoint(_write(tmp_path / "rec.csv", _recording(), sep=","))
        assert len(res["samples"]) == 19

    def test_unsorted(self, tmp_path):
        rows = _recording()
        res = pp.read_gazepoint(_write(tmp_path / "rec.tsv", rows[10:] + rows[:10]))
        assert np.all(np.diff(res["samples"]["CNT"]) > 0)
        assert list(res["events"]["label"]) == ["VAR TRIAL 1", "VAR TRIAL 2"]

    def test_without_cnt(self, tmp_path):
        """Without a usable CNT, the time is TIME; repeated times are dropped."""
        rows = _recording()
        rows = rows + [dict(rows[3], CNT=rows[4]["CNT"])]  # CNT and TIME repeated
        res = pp.read_gazepoint(_write(tmp_path / "rec.tsv", rows))
        assert res["info"]["time_base"] == "TIME"
        assert res["info"]["duplicate_times"] == 1
        np.testing.assert_array_equal(res["samples"]["time"],
                                      np.unique([r["TIME"] for r in rows]) * 1000.)

    def test_not_a_recording(self, tmp_path):
        fname = tmp_path / "other.tsv"
        fname.write_text("a\tb\n1\t2\n")
        with pytest.raises(ValueError, match="not a GazePoint recording"):
            pp.read_gazepoint(str(fname))


class TestFromGazepoint:
    def test_pupil_mm(self, fname):
        d = pp.EyeData.from_gazepoint(fname)
        assert d.name == "rec_1" and len(d) == 20  # the lost sample is filled in
        assert d.fs == pytest.approx(60., rel=1e-2)
        np.testing.assert_allclose(np.diff(d.tx), 1000. / d.fs)
        left = d.data["left_pupil"]
        # the blink is masked although LPUPILV is 1 (LPV is 0)
        assert np.all(np.isnan(left[5:9])) and np.isnan(left[12])
        assert left[0] == pytest.approx(4.8)
        assert np.sum(~np.isnan(left)) == 15
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

    def test_camera(self, fname):
        d = pp.EyeData.from_gazepoint(fname)
        assert sorted(d.aux.keys()) == sorted(
            f"{eye}_{v}" for eye in ("left", "right")
            for v in ("camx", "camy", "pupilpx", "distance"))
        assert d.aux["left_camx"][0] == pytest.approx(0.3)
        assert d.aux["right_camx"][0] == pytest.approx(0.7)
        # masked where the pupil was not found (blink) and where samples were lost
        assert np.all(np.isnan(d.aux["left_camx"][5:9])) and np.isnan(d.aux["left_camx"][12])
        assert d.aux["left_pupilpx"][0] == pytest.approx(20.)
        assert d.aux["left_distance"][0] == pytest.approx(600.)
        d = pp.EyeData.from_gazepoint(fname, pupil="px")
        assert d.aux["left_pupilmm"][0] == pytest.approx(4.8)
        assert np.isnan(d.aux["right_pupilmm"][3])  # RPUPILV == 0
        d = pp.EyeData.from_gazepoint(fname, camera=False)
        assert len(d.aux.keys()) == 0

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
        assert d.events.onsets[1] == pytest.approx(d.tx[10])

    def test_without_cnt(self, tmp_path):
        rows = [{k: v for k, v in r.items()} for r in _recording(drop=())]
        cols = [c for c in COLS if c != "CNT"]
        path = tmp_path / "rec.tsv"
        path.write_text("\n".join(["\t".join(cols)] +
                                  ["\t".join(str(r[c]) for c in cols) for r in rows]) + "\n")
        d = pp.EyeData.from_gazepoint(str(path))
        assert d.info["time_base"] == "TIME" and len(d) >= 20

    def test_save_and_load(self, fname, tmp_path):
        d = pp.EyeData.from_gazepoint(fname)
        d.write_file(str(tmp_path / "rec.pkl"))
        res = pp.EyeData.from_file(str(tmp_path / "rec.pkl"))
        assert np.allclose(res.data["left_pupil"], d.data["left_pupil"], equal_nan=True)
        assert np.allclose(res.aux["left_camx"], d.aux["left_camx"], equal_nan=True)
