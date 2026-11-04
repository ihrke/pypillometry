"""Mask layers: bit-fields per channel, registry (step 5), mask policies (step 7)."""
import numpy as np
import pytest

from pypillometry import EyeDataDict
from pypillometry.eyedata.eyedatadict import LAYER_DTYPE, MAX_LAYERS


def format2_state(d):
    """State of `d` with the events as in format 2 (the rest as given by the test)."""
    from pypillometry import state
    st = state.to_state(d)
    ev = st.pop("events")
    st.update(format_version=2, event_onsets=ev["onsets"], event_labels=ev["labels"],
              event_meta=ev["meta"])
    return st


def small_dict():
    return EyeDataDict(left_pupil=[1., 2., 3., 4., 5.], left_x=[1., 2., 3., 4., 5.])


def nan_dict():
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
        # no layers; the NaN is masked by its value
        assert not d.layers["left_pupil"].any()
        np.testing.assert_array_equal(d.mask["left_pupil"], [0, 0, 0, 0, 1])
        d.set_layer("left_pupil", "blink", [0, 1, 0, 0, 0])
        d.set_with_mask("left_pupil", np.ones(5), preserve_mask=True)
        assert d.get_layer("left_pupil", "blink").sum() == 1


class TestRecLayers:
    """Recording-wide layers (one bit-field for all keys, same registry)."""

    def test_set_get_clear(self):
        d = small_dict()
        assert d.rec_layers.dtype == LAYER_DTYPE and not d.rec_layers.any()
        d.set_layer("left_pupil", "blink", [0, 1, 0, 0, 0])
        d.set_rec_layer("excluded", [1, 1, 0, 0, 0])
        assert d.layer_names == ["blink", "excluded"]
        np.testing.assert_array_equal(d.get_rec_layer("excluded"), [1, 1, 0, 0, 0])
        d.set_rec_layer("excluded", [0, 0, 0, 0, 1], replace=True)
        np.testing.assert_array_equal(d.get_rec_layer("excluded"), [0, 0, 0, 0, 1])
        c = d.copy()
        d.clear_rec_layer("excluded")
        assert not d.get_rec_layer("excluded").any()
        assert c.get_rec_layer("excluded")[4]
        assert not EyeDataDict().get_rec_layer("excluded").any()

    def test_policy_unless_does_not_apply(self):
        import pypillometry as pp
        d = pp.EyeData(left_pupil=np.arange(6, dtype=float), sampling_rate=100)
        d.data.set_layer("left_pupil", "interpolated", [1, 1, 1, 0, 0, 0])
        d.data.set_rec_layer("excluded", [0, 1, 1, 1, 0, 0])
        assert list(np.flatnonzero(d.mask("left", "pupil"))) == [1, 2, 3]
        assert list(np.flatnonzero(d.mask("left", "pupil", policy="raw"))) == []

    def test_sample_operations(self):
        import pypillometry as pp
        d = pp.EyeData(left_pupil=np.arange(12, dtype=float), sampling_rate=100)
        d.data.set_rec_layer("excluded", np.arange(12) < 7)
        c = d.crop(30, 80, inplace=False)  # samples 3..8
        np.testing.assert_array_equal(c.data.get_rec_layer("excluded"), [1, 1, 1, 1, 0, 0])
        d._decimate(4)  # windows 4/4, 3/4, 0/4
        np.testing.assert_array_equal(d.data.get_rec_layer("excluded"), [1, 1, 0])
        d._insert_samples(np.arange(4) * 40., [0, 2, 3])
        np.testing.assert_array_equal(d.data.get_rec_layer("excluded"), [1, 0, 1, 0])

    def test_state(self):
        import pypillometry as pp
        from pypillometry import state
        d = pp.EyeData(left_pupil=np.arange(6, dtype=float), sampling_rate=100)
        d.data.set_rec_layer("excluded", [0, 1, 1, 0, 0, 0])
        r = state.from_state(state.to_state(d))
        np.testing.assert_array_equal(r.data.rec_layers, d.data.rec_layers)
        assert r.data.rec_layers.dtype == LAYER_DTYPE


class TestMaskView:
    """`EyeDataDict.mask`: the mask under the default policy."""

    def test_default_policy(self):
        d = nan_dict()
        np.testing.assert_array_equal(d.mask["left_pupil"], [0, 1, 0, 0, 0])
        d.set_layer("left_pupil", "blink", [0, 0, 1, 0, 0])
        np.testing.assert_array_equal(d.mask["left_pupil"], [0, 1, 1, 0, 0])
        d.set_layer("left_pupil", "user", [0, 0, 1, 1, 0])
        d.set_layer("left_pupil", "interpolated", [0, 0, 0, 1, 0])
        np.testing.assert_array_equal(d.mask["left_pupil"], [0, 1, 1, 0, 0])
        np.testing.assert_array_equal(d.mask["left_pupil"].astype(bool),
                                      d.get_policy_mask("left_pupil"))

    def test_read_only(self):
        d = small_dict()
        with pytest.raises(ValueError):
            d.mask["left_pupil"][0] = 1
        with pytest.raises(TypeError):
            d.mask["left_pupil"] = np.ones(5)
        assert set(d.mask) == {"left_pupil", "left_x"}
        assert "left_x" in d.mask and d.mask.get("right_x") is None

    def test_set_mask_makes_mask_equal(self):
        d = nan_dict()
        d.set_layer("left_pupil", "user", [1, 1, 0, 0, 0])
        d.set_layer("left_pupil", "blink", [1, 1, 0, 0, 0])
        d.set_layer("left_pupil", "interpolated", [0, 0, 0, 1, 0])
        d.set_mask("left_pupil", [0, 0, 1, 1, 0])
        # sample 1 is NaN: stays masked
        np.testing.assert_array_equal(d.mask["left_pupil"], [0, 1, 1, 1, 0])
        # unmasked samples are removed from the masking layers only; newly
        # masked ones are added to "user" (and are no longer "interpolated")
        np.testing.assert_array_equal(d.get_layer("left_pupil", "user"), [0, 0, 1, 1, 0])
        assert not d.get_layer("left_pupil", "blink").any()  # a masking layer too
        assert not d.get_layer("left_pupil", "interpolated").any()


class TestSampleOperations:
    """Layers move with the samples (_keep_samples, _insert_samples, _decimate)."""

    def data(self, n=12):
        import pypillometry as pp
        d = pp.EyeData(left_pupil=np.arange(n, dtype=float), sampling_rate=100)
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
        d.data.set_layer("left_pupil", "interpolated", [1, 1, 1, 1])
        d._insert_samples(np.arange(6) * 10., [0, 1, 4, 5])
        np.testing.assert_array_equal(d.data.get_layer("left_pupil", "interpolated"),
                                      [1, 1, 0, 0, 1, 1])
        # inserted samples are NaN and masked as missing (by their value)
        np.testing.assert_array_equal(d.data.mask["left_pupil"], [0, 0, 1, 1, 0, 0])
        assert np.isnan(d.data["left_pupil"][2:4]).all()

    def test_insert_samples_with_a_value(self):
        d = self.data(4)
        d._insert_samples(np.arange(6) * 10., [0, 1, 4, 5], value=0.)
        np.testing.assert_array_equal(d.data["left_pupil"], [0, 1, 0, 0, 2, 3])
        np.testing.assert_array_equal(d.data.get_layer("left_pupil", "user"),
                                      [0, 0, 1, 1, 0, 0])

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
        # values: mean of the samples that are not masked (blink and user
        # masks); all masked: mean of all samples
        np.testing.assert_allclose(d.data["left_pupil"][[0, 5]], [3., 20.5])


class TestPolicies:
    """mask(eye, var, policy) / values(eye, var, policy)."""

    def data(self):
        import pypillometry as pp
        pupil = np.arange(10, dtype=float)
        pupil[0] = np.nan
        d = pp.EyeData(left_pupil=pupil, sampling_rate=100)
        dd = d.data
        d.set_blinks(pp.Intervals([(2, 5)], "samples"))
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

    def test_new_blinks_replace_the_mask(self):
        import pypillometry as pp
        d = self.data()
        d.set_blinks(pp.Intervals([(6, 8)], "samples"))
        assert list(np.flatnonzero(d.mask("left", "pupil"))) == [0, 6, 7, 8]
        assert list(np.flatnonzero(d.mask("left", "pupil", policy=["blink"]))) == [0, 6, 7]

    def test_values(self):
        d = self.data()
        v = d.values("left", "pupil")
        assert list(np.flatnonzero(np.isnan(v))) == [0, 2, 8]
        assert v[3] == 3. and d.data["left_pupil"][2] == 2.  # stored values unchanged
        np.testing.assert_array_equal(d.values("left", "pupil", policy="raw")[1:],
                                      np.arange(1, 10))

    def test_add_mask_sets_the_user_layer(self):
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
    """Pupil-only EyeData (100 Hz, 10 s) with NaN blinks; the pupil closes and opens
    over 5 samples around them."""
    import pypillometry as pp
    n = 1000
    t = np.arange(n) * 10.
    p = 1000 + 20 * np.sin(2 * np.pi * t / 3000)
    ramp = np.linspace(0, 1, 5)
    for on, off in [(200, 215), (500, 520), (800, 812)]:
        p[on - 5:on] -= 300 * ramp
        p[off:off + 5] -= 300 * ramp[::-1]
        p[on:off] = np.nan
    return pp.EyeData(left_pupil=p, sampling_rate=100)


class TestInterpolatedLayer:

    def test_blinks_interpolate_sets_layer(self):
        d = blink_data().pupil.blinks_detect().pupil.blinks_interpolate()
        interp = d.data.get_layer("left_pupil", "interpolated")
        iv = d.get_interpolated_blinks("left", "pupil", units="samples")["left_pupil"]
        assert len(iv) == 3
        np.testing.assert_array_equal(iv.to_mask(d).astype(bool), interp)
        # interpolated samples are used by default
        assert not d.data.mask["left_pupil"][interp].any()
        assert not d.mask("left", "pupil")[interp].any()
        assert d.mask("left", "pupil", policy="strict")[interp].all()

    def test_store_as_other_variable(self):
        d = blink_data().pupil.blinks_detect()
        d.pupil.blinks_interpolate(store_as="pupilinterp")
        assert not d.data.get_layer("left_pupil", "interpolated").any()
        assert len(d.get_interpolated_blinks("left", "pupilinterp", units="samples")["left_pupilinterp"]) == 3
        np.testing.assert_array_equal(
            d.mask("left", "pupilinterp", "strict"),
            d.mask("left", "pupil") | d.data.get_layer("left_pupilinterp", "interpolated"))

    def test_interpolate_intervals_sets_layer(self):
        import pypillometry as pp
        d = blink_data()
        iv = pp.Intervals([(200, 215), (500, 520)], units="samples")
        d.interpolate_intervals(iv, store_as_suffix="_lin")
        d.interpolate_intervals(iv)
        for var in ("pupil", "pupil_lin"):
            np.testing.assert_array_equal(
                np.asarray(d.get_interpolated_blinks("left", var, units="samples")[f"left_{var}"].intervals), [(200, 215), (500, 520)])
        assert d.mask("left", "pupil", "strict")[200:215].all()
        assert not d.mask("left", "pupil")[200:215].any()

    def test_interpolate_intervals_anchors_follow_policy(self):
        import pypillometry as pp
        d = pp.EyeData(left_pupil=np.arange(100, dtype=float), sampling_rate=100)
        d.data.add_mask("left_pupil", np.s_[40:50])
        d.interpolate_intervals(pp.Intervals([(40, 50)], "samples"))
        d.data["left_pupil"][40:50] = 1000.  # an odd reconstruction
        # default: the interpolated samples are boundary points, strict: not
        for policy, near_1000 in (("default", True), ("strict", False)):
            r = d.interpolate_intervals(pp.Intervals([(50, 55)], "samples"), margin=(-15, 10),
                                        policy=policy, inplace=False)
            assert (r.data["left_pupil"][50] > 500) == near_1000


class TestBlinkLayer:

    def test_detect_sets_layer_and_mask(self):
        d = blink_data().pupil.blinks_detect()
        blink = d.data.get_layer("left_pupil", "blink")
        np.testing.assert_array_equal(d.get_blinks("left", "pupil", units="samples")["left_pupil"].to_mask(d).astype(bool), blink)
        assert d.data.mask["left_pupil"][blink].all()
        assert d.mask("left", "pupil")[blink].all()

    def test_redetection_replaces_blinks_and_mask(self):
        # finding 4.2: masks only grew
        d = blink_data()
        d.pupil.blinks_detect(strategies=["velocity"], min_duration=10)
        n1 = int(d.data.mask["left_pupil"].sum())
        d.pupil.blinks_detect(strategies=["velocity"], min_duration=2000)
        assert d.nblinks() == {"left_pupil": 0}
        nan = np.isnan(d.data["left_pupil"])
        np.testing.assert_array_equal(d.data.mask["left_pupil"].astype(bool), nan)
        assert n1 > nan.sum()
        # running it again with the first parameters gives the first result
        d.pupil.blinks_detect(strategies=["velocity"], min_duration=10)
        assert int(d.data.mask["left_pupil"].sum()) == n1

    def test_set_blinks_none_and_clear_layer_unmask(self):
        d = blink_data()
        nan = np.isnan(d.data["left_pupil"])
        for undo in (lambda d: d.set_blinks(None, eyes="left", variables="pupil"),
                     lambda d: d.clear_layer("blink")):
            d.pupil.blinks_detect()
            assert d.data.mask["left_pupil"].sum() > nan.sum()
            undo(d)
            assert len(d.get_blinks("left", "pupil", units="samples")["left_pupil"]) == 0
            np.testing.assert_array_equal(d.data.mask["left_pupil"].astype(bool), nan)

    def test_detect_with_inplace_false_returns_a_copy(self):
        # finding 4.1: the blinks were stored in a copy that was thrown away,
        # without a history entry
        orig = blink_data()
        d = orig.pupil.blinks_detect(inplace=False)
        assert d is not orig
        assert orig.nblinks() == {"left_pupil": 0}
        assert d.nblinks() == {"left_pupil": 3}
        assert d.history[-1]["funcname"] == "pupil.blinks_detect"
        assert not orig.history or orig.history[-1]["funcname"] != "pupil.blinks_detect"
        np.testing.assert_array_equal(d.data.mask["left_pupil"].astype(bool),
                                      np.isnan(d.data["left_pupil"])
                                      | d.data.get_layer("left_pupil", "blink"))

    def test_merge_close_on_runs(self):
        # finding 4.3: a blink containing another one shrank
        import pypillometry as pp
        d = blink_data()
        d.set_blinks(pp.Intervals([(100, 400), (150, 200), (420, 430)], units="samples"),
                     eyes="left", variables="pupil")
        assert d.get_blinks("left", "pupil", units="samples")["left_pupil"].intervals == [(100, 400), (420, 430)]
        merged = d.pupil.blinks_merge_close(distance=10, units="samples", inplace=False)
        assert merged.get_blinks("left", "pupil", units="samples")["left_pupil"].intervals \
            == [(100, 400), (420, 430)]
        d.pupil.blinks_merge_close(distance=30, units="samples")
        assert d.get_blinks("left", "pupil", units="samples")["left_pupil"].intervals == [(100, 430)]
        assert d.data.mask["left_pupil"][100:430].all()

    def test_format2_blinks_are_translated(self):
        # a state without layers (format 2 as written before step 5): blinks
        # and mask are translated so that removing the blinks unmasks them
        from pypillometry import state
        d = blink_data().pupil.blinks_detect()
        st = format2_state(d)
        st["data"]["mask"] = {k: np.asarray(m) for k, m in d.data.mask.items()}
        del st["data"]["layers"], st["data"]["layer_names"], st["data"]["rec_layers"]
        st["blinks"] = {"left_pupil": d.get_blinks("left", "pupil", units="samples")["left_pupil"]._to_state()}
        st["interpolated_blinks"] = {"left_pupil": None}
        e = state.from_state(st)
        np.testing.assert_array_equal(e.data.mask["left_pupil"], d.data.mask["left_pupil"])
        assert e.get_blinks("left", "pupil", units="samples")["left_pupil"].intervals == d.get_blinks("left", "pupil", units="samples")["left_pupil"].intervals
        e.clear_layer("blink")
        np.testing.assert_array_equal(e.data.mask["left_pupil"].astype(bool),
                                      np.isnan(e.data["left_pupil"]))


class TestUserLayer:

    def test_mask_intervals(self):
        import pypillometry as pp
        d = blink_data()
        d.mask_intervals(pp.Intervals([(10, 20)], units="samples"))
        np.testing.assert_array_equal(np.flatnonzero(d.data.get_layer("left_pupil", "user")),
                                      np.arange(10, 20))
        assert d.data.mask["left_pupil"][10:20].all() and d.mask("left", "pupil")[10:20].all()
        d.clear_layer("user")
        assert not d.data.mask["left_pupil"][10:20].any()

    def test_masked_array_assignment(self):
        d = blink_data().pupil.blinks_detect()
        x = d["left_pupil"]
        x.mask[30:40] = True
        d["left_pupil"] = x * 2
        np.testing.assert_array_equal(np.flatnonzero(d.data.get_layer("left_pupil", "user")),
                                      np.arange(30, 40))
        np.testing.assert_array_equal(d.data.mask["left_pupil"], np.ma.getmaskarray(x))
        assert len(d.get_blinks("left", "pupil", units="samples")["left_pupil"]) == 3  # the blinks are kept
        # unmasked samples are removed from the masks, also from the blink
        # layer (NaN stays masked)
        x.mask[:] = False
        d["left_pupil"] = x
        np.testing.assert_array_equal(d.data.mask["left_pupil"], np.isnan(x.data))
        assert not (d.data.get_layer("left_pupil", "blink") & ~np.isnan(x.data)).any()


def test_detect_returns_the_stored_blinks():
    # blinks found by the two strategies can touch; they are one blink
    from unittest import mock
    from pypillometry.signal import preproc
    d = blink_data()
    with mock.patch.object(preproc, "helper_merge_blinks",
                           return_value=np.array([[100, 109], [110, 130]])):
        d.pupil.blinks_detect(min_duration=10)
    assert d.get_blinks("left", "pupil", units="samples")["left_pupil"].intervals == [(100, 131)]


class TestGazeLayers:

    def gaze(self):
        import pypillometry as pp
        x = np.array([10., 500, 2000, 500, 500, 500, -5, 500])
        y = np.full(8, 300.)
        d = pp.EyeData(left_x=x, left_y=y, right_x=x + 1, right_y=y, sampling_rate=100,
                        inplace=False)
        d.set_experimental_setup(screen_resolution=(1000, 800))
        return d

    def test_offscreen_layer_replaced(self):
        d = self.gaze().gaze.mask_offscreen()
        expected = np.isin(np.arange(8), [2, 6])
        for key in ("left_x", "left_y", "right_x", "right_y"):
            np.testing.assert_array_equal(d.data.get_layer(key, "offscreen"), expected)
            np.testing.assert_array_equal(d.data.mask[key], expected)
        assert not d.data.get_layer("left_x", "user").any()
        # running again (smaller screen) replaces the layer
        d.set_experimental_setup(screen_resolution=(400, 800))
        e = d.gaze.mask_offscreen()
        np.testing.assert_array_equal(e.data.get_layer("left_x", "offscreen"),
                                      np.isin(np.arange(8), [1, 2, 3, 4, 5, 6, 7]))
        assert not e.clear_layer("offscreen").data.mask["left_x"].any()

    def test_divergence_layer_replaced(self):
        d = self.gaze()
        d.data.set_layer("right_x", "divergence", np.ones(8, dtype=bool))  # stale
        d.data.add_mask("left_x", np.s_[0:1])
        d["right_x"] = d.data["right_x"] + np.where(np.arange(8) == 3, 300, 0)
        e = d.gaze.mask_divergences(threshold=100, thr_type="pixel", ignore_existing_mask=False)
        expected = np.arange(8) == 3
        for key in ("left_x", "left_y", "right_x", "right_y"):
            np.testing.assert_array_equal(e.data.get_layer(key, "divergence"), expected)
        np.testing.assert_array_equal(e.data.get_layer("left_x", "user"), np.arange(8) == 0)
        assert not e.clear_layer("divergence").data.mask["right_x"].any()


def test_masked_array_assignment_checks_key_and_shape():
    import pypillometry as pp
    d = pp.EyeData(left_pupil=np.ones(5), sampling_rate=10)
    with pytest.raises(ValueError, match="eye_variable"):
        d["dist"] = np.ma.ones(5)
    with pytest.raises(ValueError, match="shape"):
        d["left_a"] = np.ma.ones(4)
    assert list(d.data.keys()) == ["left_pupil"]


class TestNoStoredMask:
    """The mask is derived from the layers only (no stored mask)."""

    def test_filter_keeps_nan_and_masked_values(self):
        import pypillometry as pp
        pupil = 1000 + np.sin(np.arange(500) / 20)
        pupil[200:220] = np.nan
        pupil[300:310] = 0.  # zero-coded blink, masked (finding 4.9)
        d = pp.EyeData(left_pupil=pupil, sampling_rate=100, inplace=False)
        d.mask_intervals(pp.Intervals([(300, 310)], "samples"), inplace=True)
        f = d.pupil.lowpass_filter(5)
        assert np.isnan(f.data["left_pupil"][200:220]).all()
        assert (f.data["left_pupil"][300:310] == 0).all()
        assert f.mask("left", "pupil")[200:220].all()
        # the zeros do not enter the filter: the neighbours stay close to the signal
        near = np.r_[290:300, 310:320]
        np.testing.assert_allclose(f.data["left_pupil"][near], pupil[near], atol=0.1)

    def test_smoothing_keeps_nan_and_masked_values(self):
        import pypillometry as pp
        pupil = 1000 + np.sin(np.arange(500) / 20)
        pupil[200:220] = np.nan  # raised an error before
        pupil[300:310] = 0.
        d = pp.EyeData(left_pupil=pupil, sampling_rate=100, inplace=False)
        d.mask_intervals(pp.Intervals([(300, 310)], "samples"), inplace=True)
        f = d.pupil.smooth_window(winsize=50)
        assert np.isnan(f.data["left_pupil"][200:220]).all()
        assert (f.data["left_pupil"][300:310] == 0).all()
        near = np.r_[290:300, 310:320]
        np.testing.assert_allclose(f.data["left_pupil"][near], pupil[near], atol=0.1)
        assert f.data.get_layer("left_pupil", "user")[300:310].all()

    def test_decimate_keeps_fully_masked_windows_masked(self):
        import pypillometry as pp
        d = pp.EyeData(left_pupil=np.arange(8, dtype=float), sampling_rate=100)
        # window 0: half user, half interpolated -> no layer covers > 1/2
        d.data.set_layer("left_pupil", "user", [1, 1, 0, 0, 0, 0, 0, 0])
        d.data.set_layer("left_pupil", "interpolated", [0, 0, 1, 1, 0, 0, 0, 0])
        # window 1: NaN set later (not a layer)
        v = d.data["left_pupil"].copy()
        v[4:8] = np.nan
        d.data.set_with_mask("left_pupil", v, preserve_mask=True)
        d._decimate(4, "strict")  # the interpolated samples are unusable too
        np.testing.assert_array_equal(d.mask("left", "pupil", "strict"), [1, 1])
        assert d.data.get_layer("left_pupil", "user")[0]
        assert d.data.get_layer("left_pupil", "interpolated")[0]
        assert np.isnan(d.data["left_pupil"][1]) and not d.data.layers["left_pupil"][1]
        assert d.data["left_pupil"][0] == 1.5  # mean of the values

    def test_decimate_uses_interpolated_samples(self):
        import pypillometry as pp
        d = pp.EyeData(left_pupil=np.arange(8, dtype=float), sampling_rate=100)
        # window 0: half user, half interpolated -> the interpolated samples are used
        d.data.set_layer("left_pupil", "user", [1, 1, 0, 0, 0, 0, 0, 0])
        d.data.set_layer("left_pupil", "interpolated", [0, 0, 1, 1, 0, 0, 0, 0])
        d._decimate(4)
        assert d.data["left_pupil"][0] == 2.5
        np.testing.assert_array_equal(d.mask("left", "pupil"), [0, 0])

    def test_state_has_no_mask_and_old_masks_are_translated(self):
        import pypillometry as pp
        from pypillometry import state
        pupil = np.arange(6, dtype=float)
        pupil[1] = np.nan
        d = pp.EyeData(left_pupil=pupil, sampling_rate=100)
        assert "mask" not in state.to_state(d)["data"]
        st = format2_state(d)
        # a state with the old mask (no layers)
        st["data"] = {"values": st["data"]["values"],
                      "mask": {"left_pupil": np.array([0, 1, 1, 0, 0, 0], dtype=bool)}}
        r = state.from_state(st)
        np.testing.assert_array_equal(r.data.mask["left_pupil"], [0, 1, 1, 0, 0, 0])
        # NaN: masked by the value, no layer
        np.testing.assert_array_equal(r.data.get_layer("left_pupil", "user"),
                                      [0, 0, 1, 0, 0, 0])
        assert r.data.layer_names == ["user"]

    def test_development_states_with_the_nan_layer(self):
        """Format-3 states written before step 7 have the layer "_nan" (NaN at
        assignment); filled samples stay masked as "user", the bit is removed."""
        import pypillometry as pp
        from pypillometry import state
        pupil = np.arange(6, dtype=float)
        pupil[[1, 2]] = np.nan
        d = pp.EyeData(left_pupil=pupil, sampling_rate=100)
        d.data.set_layer("left_pupil", "blink", [0, 0, 0, 0, 1, 0])
        st = state.to_state(d)
        dd = st["data"]
        # as written before: "_nan" registered between other layers
        dd["layer_names"] = ["_nan", "blink", "interpolated"]
        dd["values"]["left_pupil"][[1, 3]] = [10., 30.]  # 1 filled by a filter, 3 interpolated
        dd["layers"]["left_pupil"] = np.array([0, 1, 1, 1 | 4, 2, 0], dtype=LAYER_DTYPE)
        dd["rec_layers"] = np.array([0, 0, 0, 0, 4, 0], dtype=LAYER_DTYPE)
        r = state.from_state(st)
        assert r.data.layer_names == ["blink", "interpolated", "user"]
        np.testing.assert_array_equal(np.flatnonzero(r.data.get_layer("left_pupil", "user")), [1])
        np.testing.assert_array_equal(np.flatnonzero(r.data.get_layer("left_pupil", "blink")), [4])
        np.testing.assert_array_equal(
            np.flatnonzero(r.data.get_layer("left_pupil", "interpolated")), [3])
        np.testing.assert_array_equal(np.flatnonzero(r.data.get_rec_layer("interpolated")), [4])
        np.testing.assert_array_equal(r.mask("left", "pupil"), [0, 1, 1, 0, 1, 0])


class TestConsumersUsePolicy:
    """Processing and analyses read the data under a mask policy (step 7)."""

    def binocular(self):
        import pypillometry as pp
        n = 200
        left, right = np.full(n, 1000.), np.full(n, 900.)
        d = pp.EyeData(left_pupil=left, right_pupil=right, sampling_rate=100)
        d.data.add_mask("left_pupil", np.s_[50:60])
        d.interpolate_intervals(pp.Intervals([(50, 60)], "samples"), eyes="left")
        d.data.add_mask("right_pupil", np.s_[100:110])
        return d

    def test_pupil_merge_eyes(self):
        d = self.binocular()
        m = d.pupil.merge_eyes(method="both_available", inplace=False)
        assert list(np.flatnonzero(m.mask("mean", "pupil"))) == list(range(100, 110))
        s = d.pupil.merge_eyes(method="both_available", policy="strict", inplace=False)
        assert list(np.flatnonzero(s.mask("mean", "pupil"))) == list(range(50, 60)) + \
            list(range(100, 110))

    def test_get_segments(self):
        import pypillometry as pp
        d = self.binocular()
        ep = pp.Epochs([400.], (0, 300))  # samples 40-69
        seg = d.get_segments(ep, "left_pupil")
        assert not np.ma.getmaskarray(seg.data).any()
        seg = d.get_segments(ep, "left_pupil", policy="strict")
        assert list(np.flatnonzero(np.ma.getmaskarray(seg.data)[:, 0])) == list(range(10, 20))

    def test_stat_per_event(self):
        import pypillometry as pp
        d = self.binocular()
        d.data["left_pupil"][50:60] = 0.  # an odd reconstruction
        iv = pp.Intervals([(400., 700.)], "ms")  # samples 40-69
        assert d.stat_per_event(iv, eyes="left")["left_pupil"][0] == 2000. / 3
        assert d.stat_per_event(iv, eyes="left", policy="strict")["left_pupil"][0] == 1000.

    def test_quality_report(self):
        d = self.binocular()
        art = d.quality_report().to_dict(flat=True)
        assert art["artifacts.left.pupil_masked_pct"] == 0.
        assert art["artifacts.right.pupil_masked_pct"] == 5.
        art = d.quality_report(policy="strict").to_dict(flat=True)
        assert art["artifacts.left.pupil_masked_pct"] == 5.

    def test_getitem(self):
        d = self.binocular()
        assert not np.ma.getmaskarray(d["left_pupil"]).any()
        assert np.ma.getmaskarray(d["right", "pupil"])[100:110].all()

    def test_mask_as_intervals(self):
        d = self.binocular()
        assert len(d.mask_as_intervals("left", "pupil", units="samples")["left_pupil"]) == 0
        assert d.mask_as_intervals("left", "pupil", policy="strict", units="samples")["left_pupil"].intervals == [(50, 60)]
