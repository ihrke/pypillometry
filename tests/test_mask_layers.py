"""Mask layers: bit-fields per channel, registry, compat mask (step 5)."""
import numpy as np
import pytest

from pypillometry import EyeDataDict
from pypillometry.eyedata.eyedatadict import COMPAT_LAYERS, LAYER_DTYPE, MAX_LAYERS


def small_dict():
    return EyeDataDict(left_pupil=[1., np.nan, 3., 4., 5.], left_x=[1., 2., 3., 4., 5.])


class TestLayerStorage:

    def test_new_keys_have_empty_layers(self):
        d = small_dict()
        assert d.layer_names == []
        for key in d:
            assert d.layers[key].dtype == LAYER_DTYPE
            assert not d.layers[key].any()

    def test_bits_assigned_on_first_use(self):
        d = small_dict()
        assert d.layer_bits("blink") == 0  # not registered
        d.set_layer("left_pupil", "blink", [0, 1, 1, 0, 0])
        d.set_layer("left_x", "offscreen", [1, 0, 0, 0, 0])
        assert d.layer_names == ["blink", "offscreen"]
        assert d.layer_bits("blink") == 1
        assert d.layer_bits(["blink", "offscreen"]) == 3
        np.testing.assert_array_equal(d.get_layer("left_pupil", "blink"), [0, 1, 1, 0, 0])
        assert not d.get_layer("left_x", "blink").any()

    def test_set_replace_and_clear(self):
        d = small_dict()
        d.set_layer("left_pupil", "blink", [1, 1, 0, 0, 0])
        d.set_layer("left_pupil", "blink", [0, 0, 0, 1, 0])
        np.testing.assert_array_equal(d.get_layer("left_pupil", "blink"), [1, 1, 0, 1, 0])
        d.set_layer("left_pupil", "blink", [0, 0, 1, 0, 0], replace=True)
        np.testing.assert_array_equal(d.get_layer("left_pupil", "blink"), [0, 0, 1, 0, 0])
        d.set_layer("left_pupil", "user", [1, 1, 1, 1, 1])
        d.clear_layer("left_pupil", "user", where=np.s_[:2])
        np.testing.assert_array_equal(d.get_layer("left_pupil", "user"), [0, 0, 1, 1, 1])
        d.clear_layer("left_pupil", "blink")
        assert not d.get_layer("left_pupil", "blink").any()
        assert d.get_layer("left_pupil", "user").sum() == 3  # other layers kept
        d.clear_layer("left_pupil", "nosuchlayer")  # nothing to do

    def test_too_many_layers(self):
        d = small_dict()
        for i in range(MAX_LAYERS):
            d.set_layer("left_x", f"l{i}", np.zeros(5))
        with pytest.raises(ValueError, match="layers are used"):
            d.set_layer("left_x", "onemore", np.zeros(5))

    def test_copy_and_delete(self):
        d = small_dict()
        d.set_layer("left_pupil", "blink", [0, 1, 0, 0, 0])
        c = d.copy()
        c.set_layer("left_pupil", "blink", [1, 0, 0, 0, 0])
        assert d.get_layer("left_pupil", "blink").sum() == 1
        assert c.layer_names == ["blink"]
        del c["left_pupil"]
        assert "left_pupil" not in c.layers

    def test_assignment_clears_layers(self):
        d = small_dict()
        d.set_layer("left_pupil", "blink", [0, 1, 0, 0, 0])
        d["left_pupil"] = [1., 2., 3., 4., np.nan]
        assert not d.layers["left_pupil"].any()
        d.set_layer("left_pupil", "blink", [0, 1, 0, 0, 0])
        d.set_with_mask("left_pupil", np.ones(5), preserve_mask=True)
        assert d.get_layer("left_pupil", "blink").sum() == 1


class TestCompatMask:
    """`EyeDataDict.mask`: stored mask or a layer in COMPAT_LAYERS."""

    def test_nan_and_compat_layers(self):
        d = small_dict()
        np.testing.assert_array_equal(d.mask["left_pupil"], [0, 1, 0, 0, 0])
        d.set_layer("left_pupil", "blink", [0, 0, 1, 0, 0])  # not a compat layer
        np.testing.assert_array_equal(d.mask["left_pupil"], [0, 1, 0, 0, 0])
        for i, name in enumerate(COMPAT_LAYERS):
            d.set_layer("left_pupil", name, np.arange(5) == i + 2)
        np.testing.assert_array_equal(d.mask["left_pupil"], [0, 1, 1, 1, 1])

    def test_read_only(self):
        d = small_dict()
        with pytest.raises(ValueError):
            d.mask["left_pupil"][0] = 1
        with pytest.raises(TypeError):
            d.mask["left_pupil"] = np.ones(5)
        assert set(d.mask) == {"left_pupil", "left_x"}
        assert "left_x" in d.mask and d.mask.get("right_x") is None

    def test_set_mask_makes_mask_equal(self):
        d = small_dict()
        d.set_layer("left_pupil", "user", [1, 1, 0, 0, 0])
        d.set_layer("left_pupil", "blink", [1, 1, 0, 0, 0])
        d.set_mask("left_pupil", [0, 1, 1, 0, 0])
        np.testing.assert_array_equal(d.mask["left_pupil"], [0, 1, 1, 0, 0])
        # unmasked samples are removed from the compat layers only
        np.testing.assert_array_equal(d.get_layer("left_pupil", "user"), [0, 1, 0, 0, 0])
        np.testing.assert_array_equal(d.get_layer("left_pupil", "blink"), [1, 1, 0, 0, 0])


class TestSampleOperations:
    """Layers move with the samples (_keep_samples, _insert_samples, _decimate)."""

    def data(self, n=12):
        import pypillometry as pp
        d = pp.PupilData(left_pupil=np.arange(n, dtype=float), sampling_rate=100)
        return d

    def test_keep_samples(self):
        d = self.data()
        d.data.set_layer("left_pupil", "blink", np.arange(12) % 3 == 0)
        d.crop(30, 80)  # samples 3..8
        np.testing.assert_array_equal(d.data.get_layer("left_pupil", "blink"),
                                      [1, 0, 0, 1, 0, 0])
        assert d.data.layer_names == ["blink"]

    def test_insert_samples(self):
        d = self.data(4)
        d.data.set_layer("left_pupil", "blink", [1, 1, 1, 1])
        d._insert_samples(np.arange(6) * 10., [0, 1, 4, 5])
        np.testing.assert_array_equal(d.data.get_layer("left_pupil", "blink"),
                                      [1, 1, 0, 0, 1, 1])
        # inserted samples are NaN and masked as missing (stored mask)
        np.testing.assert_array_equal(d.data.mask["left_pupil"], [0, 0, 1, 1, 0, 0])

    def test_decimate_more_than_half(self):
        d = self.data(22)
        # windows of 4: [0..3] 3 of 4 set, [4..7] 2 of 4, last window [20, 21] 2 of 2
        blink = np.zeros(22, dtype=bool)
        blink[[0, 1, 2, 4, 5, 20, 21]] = True
        d.data.set_layer("left_pupil", "blink", blink)
        d.data.set_layer("left_pupil", "user", np.arange(22) == 21)  # half of the last
        d._decimate(4)
        expected = np.zeros(6, dtype=bool)
        expected[[0, 5]] = True
        np.testing.assert_array_equal(d.data.get_layer("left_pupil", "blink"), expected)
        assert not d.data.get_layer("left_pupil", "user").any()
        # values: mean of the samples that are not masked (user is a compat layer)
        np.testing.assert_allclose(d.data["left_pupil"][[0, 5]], [1.5, 20.])


class TestPolicies:
    """mask(eye, var, policy) / values(eye, var, policy)."""

    def data(self):
        import pypillometry as pp
        pupil = np.arange(10, dtype=float)
        pupil[0] = np.nan
        d = pp.PupilData(left_pupil=pupil, sampling_rate=100)
        dd = d.data
        dd.set_layer("left_pupil", "blink", np.isin(np.arange(10), [2, 3, 4]))
        dd.set_layer("left_pupil", "interpolated", np.isin(np.arange(10), [3, 4, 5]))
        dd.set_layer("left_pupil", "user", np.arange(10) == 8)
        return d

    def test_default_strict_raw(self):
        d = self.data()
        idx = lambda m: list(np.flatnonzero(m))
        # default: NaN, blink unless interpolated, user
        assert idx(d.mask("left", "pupil")) == [0, 2, 8]
        assert idx(d.mask("left_pupil", policy="strict")) == [0, 2, 3, 4, 5, 8]
        assert idx(d.mask("left", "pupil", policy="raw")) == []
        assert idx(d.mask("left", "pupil", policy=["blink"])) == [0, 2, 3, 4]
        p = pp_policy(["blink"], nan=False)
        assert idx(d.mask("left", "pupil", policy=p)) == [2, 3, 4]

    def test_values(self):
        d = self.data()
        v = d.values("left", "pupil")
        assert list(np.flatnonzero(np.isnan(v))) == [0, 2, 8]
        assert v[3] == 3. and d.data["left_pupil"][2] == 2.  # stored values unchanged
        np.testing.assert_array_equal(d.values("left", "pupil", policy="raw")[1:],
                                      np.arange(1, 10))

    def test_stored_mask_counts_like_a_layer(self):
        d = self.data()
        d.data.add_mask("left_pupil", np.s_[5:7])
        # 5 is interpolated (usable by default), 6 is not
        assert list(np.flatnonzero(d.mask("left", "pupil"))) == [0, 2, 6, 8]

    def test_errors(self):
        d = self.data()
        with pytest.raises(ValueError, match="unknown policy"):
            d.mask("left", "pupil", policy="lenient")
        with pytest.raises(ValueError, match="unknown layer"):
            d.mask("left", "pupil", policy=["blinks"])
        with pytest.raises(KeyError, match="available channels"):
            d.mask("right", "pupil")

    def test_clear_layer(self):
        d = self.data()
        c = d.clear_layer("blink", inplace=False)
        assert d.data.get_layer("left_pupil", "blink").any()
        assert not c.data.get_layer("left_pupil", "blink").any()
        assert c.history[-1]["funcname"] == "clear_layer"
        assert list(np.flatnonzero(c.mask("left", "pupil"))) == [0, 8]


def pp_policy(*args, **kwargs):
    import pypillometry as pp
    return pp.MaskPolicy(*args, **kwargs)


def blink_data():
    """PupilData (100 Hz, 10 s) with NaN blinks."""
    import pypillometry as pp
    n = 1000
    t = np.arange(n) * 10.
    p = 1000 + 20 * np.sin(2 * np.pi * t / 3000)
    for on, off in [(200, 215), (500, 520), (800, 812)]:
        p[on:off] = np.nan
    return pp.PupilData(left_pupil=p, sampling_rate=100)


class TestInterpolatedLayer:

    def test_blinks_interpolate_sets_layer(self):
        d = blink_data().pupil_blinks_detect(apply_mask=True).pupil_blinks_interpolate()
        interp = d.data.get_layer("left_pupil", "interpolated")
        iv = d.get_interpolated_blinks("left", "pupil")
        assert len(iv) == 3
        np.testing.assert_array_equal(iv.to_mask(d).astype(bool), interp)
        # interpolated samples stay masked in the old mask, are used by default
        assert d.data.mask["left_pupil"][interp].all()
        assert not d.mask("left", "pupil")[interp].any()
        assert d.mask("left", "pupil", policy="strict")[interp].all()

    def test_store_as_other_variable(self):
        d = blink_data().pupil_blinks_detect(apply_mask=True)
        d.pupil_blinks_interpolate(store_as="pupilinterp")
        assert not d.data.get_layer("left_pupil", "interpolated").any()
        assert len(d.get_interpolated_blinks("left", "pupilinterp")) == 3
        np.testing.assert_array_equal(
            d.data.mask["left_pupilinterp"],
            d.data.mask["left_pupil"] | d.data.get_layer("left_pupilinterp", "interpolated"))
