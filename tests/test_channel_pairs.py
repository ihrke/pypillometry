"""Selection of existing (eye, variable) pairs and data with incomplete pairs."""
import numpy as np
import pytest

from pypillometry import EyeDataDict


@pytest.fixture
def incomplete_dict():
    """Left eye with pupil and gaze, right eye with pupil only."""
    n = 5
    return EyeDataDict(left_pupil=np.ones(n), left_x=np.ones(n),
                       left_y=np.ones(n), right_pupil=np.ones(n))


class TestSelect:
    def test_all_pairs(self, incomplete_dict):
        assert incomplete_dict.select() == [
            ("left", "pupil"), ("left", "x"), ("left", "y"), ("right", "pupil")]
        assert incomplete_dict.select([], []) == incomplete_dict.select()

    def test_only_existing_pairs(self, incomplete_dict):
        assert incomplete_dict.select(variables=["x", "y"]) == [("left", "x"), ("left", "y")]
        assert incomplete_dict.select(eyes="right") == [("right", "pupil")]
        assert incomplete_dict.select(variables="pupil") == [("left", "pupil"), ("right", "pupil")]

    def test_order_follows_request(self, incomplete_dict):
        assert incomplete_dict.select(["right", "left"], "pupil") == [
            ("right", "pupil"), ("left", "pupil")]

    def test_missing_pair_raises(self, incomplete_dict):
        with pytest.raises(KeyError) as err:
            incomplete_dict.select(["left", "right"], ["pupil", "x"])
        msg = str(err.value)
        assert "right_x" in msg
        assert "left_pupil, left_x, left_y, right_pupil" in msg

    def test_missing_eye_raises(self, incomplete_dict):
        with pytest.raises(KeyError, match="mean"):
            incomplete_dict.select(eyes=["left", "mean"])
        # eye exists, but not with the requested variables
        with pytest.raises(KeyError, match="right"):
            incomplete_dict.select(eyes="right", variables=["x"])

    def test_missing_variable_raises(self, incomplete_dict):
        with pytest.raises(KeyError, match="baseline"):
            incomplete_dict.select(variables=["pupil", "baseline"])

    def test_empty_dict(self):
        assert EyeDataDict().select() == []


class TestEyeNameValidation:
    def test_tuple_key_with_underscore_in_eye(self):
        d = EyeDataDict()
        with pytest.raises(ValueError, match="may not contain '_'"):
            d["my_eye", "pupil"] = np.ones(3)
        with pytest.raises(ValueError, match="may not contain '_'"):
            d.set_with_mask(("my_eye", "pupil"), np.ones(3))
        assert len(d.data) == 0

    def test_string_key_splits_at_first_underscore(self):
        # a string key is never ambiguous: the eye is everything before the first '_'
        d = EyeDataDict(left_pupil_interp=np.ones(3))
        assert d.eyes == ["left"]
        assert d.variables == ["pupil_interp"]
        assert d.select("left", "pupil_interp") == [("left", "pupil_interp")]

    def test_tuple_read_with_underscore_in_eye(self):
        # "my_eye_pupil" is eye "my", variable "eye_pupil"; ("my_eye", "pupil")
        # must not silently return it
        d = EyeDataDict(my_eye_pupil=np.ones(3))
        assert d["my", "eye_pupil"] is d.data["my_eye_pupil"]
        with pytest.raises(ValueError, match="'my_eye'"):
            d["my_eye", "pupil"]

    def test_select_with_underscore_in_eye(self):
        d = EyeDataDict(my_eye_pupil=np.ones(3))
        with pytest.raises(ValueError, match="'my_eye'"):
            d.select("my_eye", "pupil")

    def test_eyedata_tuple_keys(self):
        import pypillometry as pp
        d = pp.EyeData(left_x=np.ones(3), left_y=np.ones(3), left_pupil=np.ones(3),
                       sampling_rate=10)
        with pytest.raises(ValueError, match="may not contain '_'"):
            d["my_eye", "pupil"] = np.ma.array(np.ones(3))
        with pytest.raises(ValueError, match="may not contain '_'"):
            d["my_eye", "pupil"]
        d["mean", "pupil"] = np.ones(3)
        assert "mean_pupil" in d.data


def make_incomplete_eyedata(n=1000, fs=100):
    """EyeData with left pupil and gaze, but only pupil for the right eye."""
    import pypillometry as pp
    rng = np.random.default_rng(1)
    pupil = 5 + 0.1 * np.sin(np.arange(n) / 50)
    left_pupil = pupil + rng.normal(0, 0.01, n)
    right_pupil = pupil + rng.normal(0, 0.01, n)
    for p in (left_pupil, right_pupil):
        p[200:220] = 0  # a blink
    d = pp.EyeData(left_x=500 + rng.normal(0, 5, n), left_y=400 + rng.normal(0, 5, n),
                   left_pupil=left_pupil, right_pupil=right_pupil,
                   sampling_rate=fs, event_onsets=[1000, 5000], event_labels=["a", "b"],
                   experimental_setup=pp.ExperimentalSetup(
                       screen_resolution=(1000, 800), physical_screen_size=("50 cm", "40 cm"),
                       eye_to_screen_center="60 cm",
                       camera_spherical=("20 deg", "-90 deg", "60 cm")))
    return d


@pytest.fixture
def incomplete_eyedata():
    return make_incomplete_eyedata()


class TestIncompleteBlinkStores:
    def test_blinks_initialised_for_existing_pairs(self, incomplete_eyedata):
        d = incomplete_eyedata
        assert set(d._blinks) == {"left_pupil", "left_x", "left_y", "right_pupil"}

    def test_set_and_get_blinks(self, incomplete_eyedata):
        import pypillometry as pp
        d = incomplete_eyedata
        iv = pp.Intervals([(10, 20)], units="samples", data_time_range=(0, len(d)))
        d.set_blinks(iv)
        assert set(d.get_blinks()) == {"left_pupil", "left_x", "left_y", "right_pupil"}
        assert d.data.mask["right_pupil"][10:20].all()
        assert d.nblinks() == {"left_pupil": 1, "left_x": 1, "left_y": 1, "right_pupil": 1}
        assert set(d.get_blinks(variables="x")) == {"left_x"}
        with pytest.raises(KeyError, match="right_x"):
            d.get_blinks("right", "x")
        with pytest.raises(KeyError, match="right_x"):
            d.set_blinks(iv, eyes="right", variables="x")

    def test_mask_as_intervals(self, incomplete_eyedata):
        d = incomplete_eyedata
        assert set(d.mask_as_intervals()) == {"left_pupil", "left_x", "left_y", "right_pupil"}
        with pytest.raises(KeyError, match="available channels"):
            d.mask_as_intervals("right", "x")

    def test_blink_stats(self, incomplete_eyedata):
        d = incomplete_eyedata
        d.pupil_blinks_detect()
        d["mean", "x"] = d.data["left_x"]  # an eye without pupil data
        stats = d.blink_stats()
        assert stats["mean"] is None
        assert stats["left"]["n"] >= 1 and stats["right"]["n"] >= 1


class TestIncompleteScaleMerge:
    def test_scale_unscale(self, incomplete_eyedata):
        d = incomplete_eyedata
        orig = {k: v.copy() for k, v in d.data.data.items()}
        d.scale()
        assert set(d.params["scale"]["mean"]) == {"left", "right"}
        assert set(d.params["scale"]["mean"]["right"]) == {"pupil"}
        np.testing.assert_allclose(np.nanmean(d.data["right_pupil"][d.data.mask["right_pupil"] == 0]),
                                   0, atol=1e-10)
        d.unscale()
        for k, v in orig.items():
            np.testing.assert_allclose(d.data[k], v)
        with pytest.raises(KeyError, match="right_x"):
            d.scale(eyes=["left", "right"], variables=["pupil", "x"])

    def test_merge_eyes(self, incomplete_eyedata):
        d = incomplete_eyedata
        d.merge_eyes()
        np.testing.assert_allclose(d.data["mean_pupil"],
                                   (d.data["left_pupil"] + d.data["right_pupil"]) / 2)
        np.testing.assert_allclose(d.data["mean_x"], d.data["left_x"])

    def test_merge_eyes_drop_eyes(self, incomplete_eyedata):
        d = incomplete_eyedata
        d.merge_eyes(variables="pupil", keep_eyes=False)
        assert set(d.data.keys()) == {"left_x", "left_y", "mean_pupil"}
        assert d._blinks.get("right_pupil") is None

    def test_scale_after_merge_eyes(self, incomplete_eyedata):
        # used to fail with KeyError: 'mean_x' (D6)
        d = incomplete_eyedata
        d.merge_eyes(variables="pupil")
        d.scale()
        assert set(d.params["scale"]["mean"]["mean"]) == {"pupil"}


class TestIncompleteIntervals:
    def test_mask_and_interpolate_intervals(self, incomplete_eyedata):
        import pypillometry as pp
        d = incomplete_eyedata
        iv = pp.Intervals([(100, 110)], units="samples", data_time_range=(0, len(d)))
        d.mask_intervals(iv)
        for key in ["left_pupil", "left_x", "left_y", "right_pupil"]:
            assert d.data.mask[key][100:110].all()
        d.interpolate_intervals(iv, store_as_suffix="_int")
        assert set(d.data.keys()) >= {"left_x_int", "left_y_int", "left_pupil_int",
                                      "right_pupil_int"}
        with pytest.raises(KeyError, match="right_x"):
            d.mask_intervals(iv, eyes="right", variables="x")
        with pytest.raises(KeyError, match="right_x"):
            d.interpolate_intervals(iv, eyes=["left", "right"], variables="x")

    def test_blinks_merge_close(self, incomplete_eyedata):
        import pypillometry as pp
        d = incomplete_eyedata
        iv = pp.Intervals([(10, 20), (22, 30)], units="samples", data_time_range=(0, len(d)))
        d.set_blinks(iv)
        merged = d.blinks_merge_close(distance=5, units="samples")
        assert {k: len(v) for k, v in merged.items()} == {
            "left_pupil": 1, "left_x": 1, "left_y": 1, "right_pupil": 1}
        d.blinks_merge_close(distance=5, units="samples", apply_mask=True)
        assert d.nblinks() == {"left_pupil": 1, "left_x": 1, "left_y": 1, "right_pupil": 1}

    def test_stat_per_event(self, incomplete_eyedata):
        import pypillometry as pp
        d = incomplete_eyedata
        iv = pp.Intervals([(100, 200), (300, 400)], units="samples", data_time_range=(0, len(d)))
        stat = d.stat_per_event(iv)
        assert set(stat) == {"left_pupil", "left_x", "left_y", "right_pupil"}
        assert stat["right_pupil"].shape == (2,)
        with pytest.raises(KeyError, match="right_y"):
            d.stat_per_event(iv, eyes="right", variables="y")


class TestIncompletePupilMethods:
    @pytest.fixture
    def d(self, incomplete_eyedata):
        d = incomplete_eyedata
        d["extra", "x"] = d.data["left_x"]  # an eye without pupil data
        return d

    def test_filters_skip_eyes_without_pupil(self, d):
        d.pupil_lowpass_filter(cutoff=5)
        d.pupil_smooth_window(winsize=50)
        with pytest.raises(KeyError, match="extra"):
            d.pupil_lowpass_filter(cutoff=5, eyes="extra")

    def test_blinks(self, d):
        d.pupil_blinks_detect()
        nb = d.nblinks()
        assert set(nb) == {"left_pupil", "right_pupil"}
        d.pupil_blinks_interpolate()
        assert set(d.get_interpolated_blinks(variables="pupil")) == {"left_pupil", "right_pupil"}
        assert np.all(d.data["right_pupil"][200:220] > 4)


class TestIncompleteGazeMethods:
    def test_correct_pupil_foreshortening(self, incomplete_eyedata):
        d = incomplete_eyedata
        right = d.data["right_pupil"].copy()
        d.correct_pupil_foreshortening(store_as="pupilc")
        assert "left_pupilc" in d.data and "right_pupilc" not in d.data
        np.testing.assert_array_equal(d.data["right_pupil"], right)
        with pytest.raises(KeyError, match="right_x"):
            d.correct_pupil_foreshortening(eyes="right")

    def test_mask_offscreen_coords(self, incomplete_eyedata):
        d = incomplete_eyedata
        d.data["left_x"][50:60] = 5000
        d.mask_offscreen_coords()
        assert d.data.mask["left_x"][50:60].all()
        with pytest.raises(KeyError, match="right"):
            d.mask_offscreen_coords(eyes="right")


class TestIncompletePlots:
    @pytest.fixture(autouse=True)
    def agg(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def test_plot_intervals(self, incomplete_eyedata):
        import pypillometry as pp
        d = incomplete_eyedata
        iv = pp.Intervals([(100, 200), (300, 400)], units="samples", data_time_range=(0, len(d)))
        figs = d.plot.plot_intervals(iv, show_mask=True)
        labels = [line.get_label() for line in figs[0].axes[0].get_lines()]
        assert labels == ["left_x", "left_y", "left_pupil", "right_pupil"]

    def test_plot_timeseries(self, incomplete_eyedata):
        import matplotlib.pyplot as plt
        d = incomplete_eyedata
        plt.figure()
        d.plot.plot_timeseries(variables=["pupil", "x"])
        assert len(plt.gcf().axes) == 2
        with pytest.raises(KeyError, match="right_x"):
            d.plot.plot_timeseries(eyes=["left", "right"], variables=["x"])

    def test_plot_masked(self, incomplete_eyedata):
        import matplotlib.pyplot as plt
        d = incomplete_eyedata
        d.data.mask["right_pupil"][10:20] = 1
        plt.figure()
        d.plot.plot_masked("right", "pupil")
        d.plot.plot_masked()
