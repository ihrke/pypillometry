"""Transplantation of blinks into clean data (pypillometry.signal.transplant)."""
import numpy as np
import pytest

from pypillometry.signal import transplant as T


def make_signal(n=60000, nblinks=40, seed=0, core=60, missing=np.nan):
    """Smooth signal with blinks (drop over 20, recovery over 40 samples)."""
    rng = np.random.default_rng(seed)
    clean = 1000 + np.convolve(rng.normal(0, 30, n + 400), np.hanning(401) / 200, "valid")
    clean += rng.normal(0, 1, n)
    y = clean.copy()
    starts = np.linspace(500, n - 1000, nblinks).astype(int) + rng.integers(0, 200, nblinks)
    for s in starts:
        amp = rng.uniform(300, 500)
        y[s - 20:s] -= np.linspace(0, amp, 20)
        y[s + core:s + core + 40] -= np.linspace(amp, 0, 40)
        y[s:s + core] = missing
    return y, clean, np.c_[starts, starts + core]


class TestCores:
    def test_runs_of_missing_samples(self):
        y = np.ones(100)
        y[10:20] = np.nan
        y[30:35] = 0
        np.testing.assert_array_equal(T.blink_cores(y), [[10, 20], [30, 35]])
        np.testing.assert_array_equal(T.blink_cores(y, blink_val=np.nan), [[10, 20]])
        assert T.blink_cores(np.ones(10)).shape == (0, 2)

    def test_merge_distance(self):
        y = np.ones(100)
        y[10:20] = y[25:30] = y[50:60] = np.nan
        np.testing.assert_array_equal(T.blink_cores(y, 4), [[10, 20], [25, 30], [50, 60]])
        np.testing.assert_array_equal(T.blink_cores(y, 5), [[10, 30], [50, 60]])
        np.testing.assert_array_equal(T.blink_cores(y, 50), [[10, 60]])

    def test_donors_need_signal_around_them(self):
        y = np.ones(1000)
        for a, b in [(5, 15), (300, 320), (350, 360), (600, 640), (990, 1000)]:
            y[a:b] = np.nan
        cores = T.blink_cores(y)
        np.testing.assert_array_equal(T.blink_donors(cores, np.isnan(y), (20, 20)),
                                      [False, True, True, True, False])
        # the neighbour is in the window
        np.testing.assert_array_equal(T.blink_donors(cores, np.isnan(y), (20, 40)),
                                      [False, False, True, True, False])
        np.testing.assert_array_equal(T.blink_donors(cores, np.isnan(y), 20, min_len=15, max_len=30),
                                      [False, True, False, False, False])


class TestArtifacts:
    def test_baseline_removed_and_tapered(self):
        y = 500 + 2.0 * np.arange(400)          # slow baseline
        y[180:200] -= np.linspace(0, 100, 20)
        y[200:220] = np.nan
        art, = T.extract_blink_artifacts(y, [(200, 220)], (100, 80), edge=20)
        assert len(art) == 200
        assert np.isnan(art[100:120]).all() and not np.isnan(art[:100]).any()
        np.testing.assert_allclose(art[:80], 0, atol=1e-9)
        np.testing.assert_allclose(art[120:], 0, atol=1e-9)
        np.testing.assert_allclose(art[80:100], -np.linspace(0, 100, 20))

    def test_multiplicative(self):
        y = np.full(400, 800.)
        y[180:200] *= np.linspace(1, 0.5, 20)
        y[200:220] = 0
        art, = T.extract_blink_artifacts(y, [(200, 220)], 100, mode="multiplicative")
        np.testing.assert_allclose(art[:80], 1)
        np.testing.assert_allclose(art[80:100], np.linspace(1, 0.5, 20))
        out = T.insert_blink_artifacts(np.full(300, 400.), [art], [(50, 0)], mode="multiplicative")
        np.testing.assert_allclose(out[130:150], 400 * np.linspace(1, 0.5, 20))
        with pytest.raises(ValueError):
            T.extract_blink_artifacts(y - 900, [(200, 220)], 100, mode="multiplicative")

    def test_window_outside_data(self):
        with pytest.raises(ValueError):
            T.extract_blink_artifacts(np.ones(100), [(10, 20)], 50)

    def test_insert(self):
        art = np.r_[0., -1, -2, np.nan, np.nan, -1, 0]
        out = T.insert_blink_artifacts(np.full(20, 10.), [art], [(2, 0), (12, 0)], missing_value=0)
        np.testing.assert_array_equal(out[2:9], [10, 9, 8, 0, 0, 9, 10])
        np.testing.assert_array_equal(out[12:19], out[2:9])
        np.testing.assert_array_equal(out[[0, 1, 9, 10, 11, 19]], 10)
        with pytest.raises(ValueError):
            T.insert_blink_artifacts(np.ones(20), [art], [(15, 0)])

    def test_shrink_removes_fluctuations(self):
        """far from the core, the artifacts become the median artifact"""
        y, clean, cores = make_signal()
        arts = T.extract_blink_artifacts(y, cores, (150, 250))
        null = T.extract_blink_artifacts(clean, cores + 400, (150, 250))
        profile = T.blink_artifact_profile(arts, null, (150, 250))
        (w0, w1), (m0, m1) = profile["weight"], profile["median"]
        assert len(w0) == len(m0) == 150 and len(w1) == len(m1) == 250
        assert w0[-5:].min() > 0.8 and w1[:10].min() > 0.8     # the blinks differ
        assert w0[:100].max() < 0.5 and w1[100:].max() < 0.5   # only fluctuations
        assert (np.diff(w0) >= 0).all() and (np.diff(w1) <= 0).all()
        assert m0[-1] < -200 and abs(m0[:100]).max() < 5
        shrunk = T.shrink_blink_artifacts(arts, profile)
        far = lambda aa: np.sqrt(np.mean([a[:100] ** 2 for a in aa]))
        assert far(shrunk) < 0.5 * far(arts)


class TestPlaces:
    def test_clean_stretches(self):
        missing = np.zeros(1000, dtype=bool)
        missing[100:110] = missing[500:520] = True
        np.testing.assert_array_equal(T.clean_stretches(missing, (20, 50)),
                                      [[0, 80], [160, 480], [570, 1000]])
        np.testing.assert_array_equal(T.clean_stretches(missing, (20, 50), min_len=100),
                                      [[160, 480], [570, 1000]])

    def test_places_do_not_overlap(self):
        stretches = np.array([[0, 1000], [2000, 2300], [5000, 9000]])
        lengths = [200, 250, 300]
        places = T.place_blink_artifacts(stretches, lengths, pad=30, seed=1)
        start, end = places[:, 0], places[:, 0] + np.take(lengths, places[:, 1])
        assert (start[1:] - end[:-1] >= 60).all()
        inside = [((start - 30 >= a) & (end + 30 <= b)) for a, b in stretches]
        assert np.any(inside, axis=0).all()
        assert set(places[:, 1]) == {0, 1, 2}
        np.testing.assert_array_equal(places, T.place_blink_artifacts(stretches, lengths, pad=30, seed=1))
        assert len(T.place_blink_artifacts(stretches, lengths, n=4, pad=30, seed=1)) == 4
        assert len(T.place_blink_artifacts(stretches, [])) == 0


class TestTransplant:
    def test_transplanted_signal(self):
        y, clean, cores = make_signal()
        tr = T.transplant_blinks(y, (150, 250), pad=50, seed=3)
        assert len(tr) > 20 and len(tr.donors) == len(cores)
        np.testing.assert_array_equal(tr.clean, y)
        # changed only in the windows, which are in clean signal
        changed = ~((tr.signal == y) | (np.isnan(tr.signal) & np.isnan(y)))
        inwin = np.zeros(len(y), dtype=bool)
        for (a, b), (c0, c1), (s0, s1) in zip(tr.windows, tr.blinks, tr.sites):
            inwin[a:b] = True
            assert not np.isnan(y[s0:s1]).any()
            assert np.isnan(tr.signal[c0:c1]).all()
            assert (a, b) == (c0 - 150, c1 + 250) and (s0, s1) == (a - 50, b + 50)
        assert not changed[~inwin].any()
        # the blinks look like the original ones
        drop = np.array([tr.signal[c0 - 1] - tr.clean[c0 - 1] for c0, _ in tr.blinks])
        assert (drop < -200).all()

    def test_error(self):
        y, clean, cores = make_signal()
        tr = T.transplant_blinks(y, (150, 250), pad=50, seed=3)
        assert tr.error(tr.clean) == 0
        raw = tr.error(tr.signal)                    # cores: interpolated linearly
        assert raw > 10
        per_site = tr.error(tr.signal, per_site=True)
        assert per_site.shape == (len(tr),)
        size = tr.sites[:, 1] - tr.sites[:, 0]
        assert np.isclose(np.sqrt((per_site ** 2 * size).sum() / size.sum()), raw)
        assert np.isnan(T.transplant_error(y, y, np.zeros((0, 2))))

    def test_n_seed_and_subset(self):
        y, clean, cores = make_signal()
        a = T.transplant_blinks(y, (150, 250), n=7, seed=5)
        b = T.transplant_blinks(y, (150, 250), n=7, seed=5)
        assert len(a) == 7
        np.testing.assert_array_equal(a.signal, b.signal)
        sub = T.transplant_blinks(y, (150, 250), cores=cores[:10], seed=5)
        np.testing.assert_array_equal(sub.donors, cores[:10])

    def test_missing_value_code_is_kept(self):
        y, clean, cores = make_signal(missing=0)
        tr = T.transplant_blinks(y, (150, 250), n=5, seed=1)
        assert not np.isnan(tr.signal).any()
        assert all((tr.signal[a:b] == 0).all() for a, b in tr.blinks)

    def test_masked_samples_are_avoided(self):
        y, clean, cores = make_signal()
        masked = np.zeros(len(y), dtype=bool)
        masked[:30000] = True
        tr = T.transplant_blinks(y, (150, 250), missing=masked, seed=1)
        assert tr.sites.min() >= 30000 and tr.donors.min() >= 30000

    def test_into_other_signal(self):
        y, clean, cores = make_signal()
        target = np.full(20000, 5.0)
        tr = T.transplant_blinks(target, (150, 250), source=y, seed=1)
        assert len(tr) > 10 and len(tr.signal) == 20000
        assert np.nanmin(tr.signal) < -200

    def test_no_blinks(self):
        tr = T.transplant_blinks(np.ones(1000), (50, 50))
        assert len(tr) == 0 and tr.sites.shape == (0, 2)
        np.testing.assert_array_equal(tr.signal, 1)


def make_mixed(n=40000, seed=0):
    """Smooth signal with blinks (drop over 20 samples, 60 missing, recovery
    over 40), tracker loss (60 missing, no transients), a blink with
    another run of missing samples close to it, and fast, short dips
    without missing samples."""
    rng = np.random.default_rng(seed)
    y = 1000 + np.convolve(rng.normal(0, 30, n + 400), np.hanning(401) / 200, "valid")
    y += rng.normal(0, 1, n)
    slots = np.arange(400, n - 400, 400)
    kinds = np.resize(["blink", "dip", "blink", "loss", "blink", "dip", "blink", "double"],
                      len(slots))
    for s, kind in zip(slots, kinds):
        if kind in ("blink", "double"):
            amp = rng.uniform(300, 500)
            y[s - 20:s] -= np.linspace(0, amp, 20)
            y[s + 60:s + 100] -= np.linspace(amp, 0, 40)
            y[s:s + 60] = np.nan
            if kind == "double":
                y[s + 130:s + 150] = np.nan
        elif kind == "loss":
            y[s:s + 60] = np.nan
        else:
            y[s:s + 4] -= np.linspace(0, 150, 4)
            y[s + 4:s + 8] -= np.linspace(150, 0, 4)
    return y, slots, kinds


WINDOW = (50, 100)


class TestObviousBlinks:
    @pytest.fixture(scope="class")
    def mixed(self):
        y, slots, kinds = make_mixed()
        return y, slots, kinds, T.obvious_blinks(y, WINDOW, min_len=10, max_len=250)

    def test_only_blinks_with_transients(self, mixed):
        y, slots, kinds, ob = mixed
        np.testing.assert_array_equal(ob.blinks[:, 0], slots[kinds == "blink"])
        assert len(ob) == (kinds == "blink").sum()
        assert len(ob.cores) == (kinds != "dip").sum() + (kinds == "double").sum()
        # tracker loss is isolated, but has no transients; the double blink is not isolated
        loss = np.isin(ob.cores[:, 0], slots[kinds == "loss"])
        double = np.isin(ob.cores[:, 0], slots[kinds == "double"])
        assert ob.isolated[loss].all() and not ob.obvious[loss].any()
        assert not ob.isolated[double].any() and np.isnan(ob.drop[double]).all()
        assert (ob.drop[ob.obvious] > ob.drop_threshold).all() and ob.drop_threshold > 0
        assert (ob.rise[ob.obvious] > ob.rise_threshold).all() and ob.rise_threshold > 0
        assert "obvious" in repr(ob)

    def test_lengths_and_other_missing_samples(self, mixed):
        y, slots, kinds, ob = mixed
        assert len(T.obvious_blinks(y, WINDOW, min_len=61)) == 0
        assert len(T.obvious_blinks(y, WINDOW, max_len=59)) == 0
        masked = np.zeros(len(y), dtype=bool)
        first = ob.blinks[0, 0]
        masked[first - 10] = True      # in the window of the first blink
        other = T.obvious_blinks(y, WINDOW, missing=masked)
        assert first not in other.blinks[:, 0] and len(other) == len(ob) - 1

    def test_given_cores_and_avoid(self, mixed):
        y, slots, kinds, ob = mixed
        some = ob.cores[[0, 1, 2]]
        given = T.obvious_blinks(y, WINDOW, cores=some)
        np.testing.assert_array_equal(given.cores, some)
        np.testing.assert_array_equal(given.obvious, ob.obvious[:3])
        # the dips are not used for the thresholds when avoided
        dips = np.zeros(len(y), dtype=bool)
        for s in slots[kinds == "dip"]:
            dips[s:s + 8] = True
        avoided = T.obvious_blinks(y, WINDOW, avoid=dips)
        assert avoided.drop_threshold < ob.drop_threshold

    def test_errors(self, mixed):
        y = mixed[0]
        with pytest.raises(ValueError):
            T.obvious_blinks(np.r_[np.ones(50), np.nan, np.ones(50)], WINDOW)   # no clean signal
        with pytest.raises(ValueError):
            T.obvious_blinks(y, WINDOW, edge=60)
        with pytest.raises(ValueError):
            T.obvious_blinks(y, WINDOW, quantile=1)
