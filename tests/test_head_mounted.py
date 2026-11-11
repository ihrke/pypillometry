"""Tests for head-mounted eye trackers (HeadMountedEyeData, Tobii Pro Lab exports).

`tests/data/tobii_prolab.tsv` is a small synthetic export: eye tracker at
100 Hz from 1005 to 1495 ms with a gap (1200-1250 ms) and invalid samples
(1300-1330 ms), gyroscope (3, 4, 0), accelerometer, magnetometer and four
events.
"""
import os

import numpy as np
import pytest

import pypillometry as pp

FNAME = os.path.join(os.path.dirname(__file__), "data", "tobii_prolab.tsv")


@pytest.fixture
def data():
    return pp.HeadMountedEyeData.from_tobii_prolab(FNAME)


def _write_video(fname, n=50, size=(64, 36), fps=100):
    """Video with `n` frames: the first half black, the second half white."""
    import av
    with av.open(str(fname), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n):
            img = np.full((size[1], size[0], 3), 0 if i < n // 2 else 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class TestReader:
    def test_tables(self):
        tabs = pp.read_tobii_prolab(FNAME)
        assert len(tabs["eye"]) == 45 and len(tabs["mag"]) == 5
        assert list(tabs["gyro"].columns) == ["time", "x", "y", "z"]
        assert list(tabs["events"]["label"]) == ["Start", "SyncPortOutHigh", "stimulus", "Stopp"]
        assert tabs["info"]["participant"] == "P01"

    def test_not_an_export(self, tmp_path):
        fname = tmp_path / "other.tsv"
        fname.write_text("a\tb\n1\t2\n")
        with pytest.raises(ValueError, match="not a Tobii Pro Lab data export"):
            pp.read_tobii_prolab(fname)


class TestFromTobii:
    def test_channels(self, data):
        assert isinstance(data, pp.HeadMountedEyeData) and data.name == "rec1"
        assert list(data.data) == ["left_pupil", "right_pupil", "both_x", "both_y"]
        assert data.fs == 100 and len(data) == 50  # the gap is filled in
        assert data.scene_size == (64, 36) and data.scene_video is None
        assert data.scene_video_offset == 1000
        # the gap and the invalid samples are missing
        missing = ((data.tx >= 1200) & (data.tx < 1250)) | ((data.tx >= 1300) & (data.tx < 1330))
        np.testing.assert_array_equal(data.mask("left", "pupil"), missing)
        np.testing.assert_array_equal(data.mask("both", "x"), missing)

    def test_aux(self, data):
        assert {"gaze3d_z", "gyro_x", "acc_y", "mag_z"} <= set(data.aux)
        assert all(len(v) == len(data) for v in data.aux.values())
        # the motion sensors have data in the gap of the eye tracker
        assert np.allclose(data.aux["gyro_x"], 3)

    def test_events(self, data):
        assert list(data.events.labels) == ["Start", "stimulus", "Stopp"]
        # events outside the samples are moved to the first/last sample
        np.testing.assert_array_equal(data.events.onsets, [1005, 1400, 1495])
        with_sync = pp.HeadMountedEyeData.from_tobii_prolab(FNAME, remove_sync_events=False)
        assert "SyncPortOutHigh" in with_sync.events.labels

    def test_aux_follows_the_samples(self, data):
        assert len(data.crop(1100, 1300, inplace=False).aux["gyro_x"]) == 20
        assert len(data.downsample(50, inplace=False).aux["gyro_x"]) == 25

    def test_repr_html(self, data):
        pytest.importorskip("jinja2")
        html = data._repr_html_()
        assert "Aux channels" in html and "gyro_x" in html and "Additional channels (aux)" in html
        assert "<figure" not in html  # no scene video

    def test_save_and_load(self, data, tmp_path):
        for name in ("d.pypil", "d.h5"):
            data.write_file(str(tmp_path / name))
            res = pp.EyeData.from_file(str(tmp_path / name))
            assert isinstance(res, pp.HeadMountedEyeData)
            assert res.scene_size == (64, 36) and res.scene_video_offset == 1000
            np.testing.assert_array_equal(res.aux["acc_z"], data.aux["acc_z"])


class TestPlots:
    def test_timeseries_with_aux(self, data):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = data.plot.plot_timeseries(variables=["pupil", "gyro", "acc_x"])
        assert [ax.get_title() for ax in fig.axes] == ["pupil", "gyro", "acc_x"]
        assert [line.get_label() for line in fig.axes[1].lines] == ["gyro_x", "gyro_y", "gyro_z"]
        assert fig.axes[0].get_xlabel() == "" and fig.axes[2].get_xlabel() == "sec"
        assert len(data.plot.plot_timeseries().axes) == 3  # pupil, x, y: no aux
        with pytest.raises(ValueError, match="no variable or aux channel nothing"):
            data.plot.plot_timeseries(variables=["pupil", "nothing"])
        plt.close("all")

    def test_pupil_covariate(self, data):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        assert data.plot.plot_pupil_covariate("gaze3d_z", log=True, ax=ax) is ax
        labels = ax.get_legend_handles_labels()[1]
        assert len(labels) == 2 and labels[0].startswith("left (r = ")
        assert ax.get_xlabel() == "gaze3d_z" and ax.get_xscale() == "log"
        with pytest.raises(ValueError, match=r"no channel\(s\) nothing in aux"):
            data.plot.plot_pupil_covariate("nothing")
        plt.close(fig)


class TestScene:
    def test_distance_and_head_movement(self, data):
        res = data.scene.distance(window=0).scene.head_movement()
        assert res is data
        ok = ~np.isnan(data.aux["gaze3d_z"])
        expected = np.sqrt(10 ** 2 + 20 ** 2 + data.aux["gaze3d_z"][ok] ** 2)
        np.testing.assert_allclose(data.aux["gaze_distance"][ok], expected)
        assert np.isnan(data.aux["gaze_distance"][~ok]).all()
        np.testing.assert_allclose(data.aux["head_speed"], 5)
        assert [h["funcname"] for h in data.history] == ["scene.distance", "scene.head_movement"]

    def test_pipeline_step(self, data):
        res = data | pp.scene.head_movement()
        assert "head_speed" in res.aux and "head_speed" not in data.aux

    def test_missing_video(self, data):
        pytest.importorskip("av")
        with pytest.raises(ValueError, match="no scene video"):
            data.scene.luminance()

    def test_missing_channels(self, data):
        del data.aux["gyro_x"]
        with pytest.raises(ValueError, match=r"no channel\(s\) gyro_x in aux"):
            data.scene.head_movement()

    def test_video(self, data, tmp_path):
        av = pytest.importorskip("av")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data.scene_video = str(tmp_path / "scene.mp4")
        _write_video(data.scene_video)  # 0.5 s, starts at 1000 ms

        assert data.scene.frame(1100).shape == (36, 64, 3)
        assert data.scene.frame(1100).mean() < 30 and data.scene.frame("1.4 sec").mean() > 225

        data.scene.luminance(radius=5)
        lum = data.aux["scene_luminance"]
        assert lum[data.tx < 1240].max() < 0.05 and lum[data.tx > 1260].min() > 0.9
        # no gaze position: no luminance at the gaze position
        np.testing.assert_array_equal(np.isnan(data.aux["gaze_luminance"]), data.mask("both", "x"))

        fig, ax = plt.subplots()
        assert data.plot.plot_scene_scanpath((1.1, 1.2), title="T", ax=ax) is ax
        assert len(ax.images) == 1 and ax.get_title() == "T"
        assert ax.get_xlim() == (0, 64) and ax.get_ylim() == (36, 0)
        plt.close(fig)

        pytest.importorskip("jinja2")
        html = data._repr_html_()
        assert "data:image/jpeg;base64," in html and "scene.mp4 (64x36)" in html

        # video with the gaze position: 1.1 to 1.2 s are 10 frames, red at the gaze
        out = data.scene.render_video(str(tmp_path / "gaze.mp4"), (1.1, 1.2), radius=3)
        from pypillometry import _video
        assert _video._video_info(out)["size"] == (64, 36)
        with av.open(out) as container:
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        assert len(frames) == 10
        i = np.argmin(np.abs(data.tx - 1105))
        x, y = int(data.data["both_x"][i]), int(data.data["both_y"][i])
        r, g, b = frames[0][y, x - 3].astype(int)
        assert r > g + 80 and r > b + 80
        # strip with the pupil size at the bottom: white line in its middle
        assert frames[0][32, 32].mean() > 100 and frames[0][32, 10].mean() < 60
        out = data.scene.render_video(out, (1.1, 1.2), pupil=False)
        with av.open(out) as container:
            assert next(container.decode(video=0)).to_ndarray(format="rgb24")[32, 32].mean() < 60
        # preview: every fifth frame, limited number of threads
        out = data.scene.render_video(out, (1.1, 1.2), fps=20, threads=1)
        with av.open(out) as container:
            assert len(list(container.decode(video=0))) == 2
            assert float(container.streams.video[0].average_rate) == 20
        with pytest.raises(ValueError, match="threads must be"):
            data.scene.luminance(threads=-1)
        # only the part of the video with samples is written
        part = data.crop(1100, 1300, inplace=False)
        assert _video._video_info(part.scene.render_video(out))["duration"] == pytest.approx(0.2, abs=0.02)
        with pytest.raises(ValueError, match="does not overlap with the data"):
            part.scene.render_video(out, (1.35, 1.45))
