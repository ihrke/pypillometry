"""pp.TimescaleModel and pp.TimescaleGroup on simulated recordings."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pypillometry as pp
from pypillometry.signal import gaussmarkov as gm

FS = 100.0          # of the recordings; the models run at 20 Hz
TS = ("3 s", "30 s")


def recording(seed=1, minutes=5, name="rec", inplace=False, gaps=True):
    """Binocular recording at 100 Hz with blinks (0) and a long gap; returns
    the data and the true noise-free pupil size of each eye."""
    bank = gm.TimescaleBank(timescales=(3000., 30000.), channels=("left", "right"))
    theta = bank.theta(fast_sd=8., fast_ell=700., noise_sd_left=0.01, noise_sd_right=0.01,
                       diff_sd=2., diff_ell=600., gain=1., sd_3s=3., sd_30s=5.)
    n = int(minutes * 60 * FS)
    _, x = bank.simulate(theta, n, FS, rng=seed, return_states=True)
    rng = np.random.default_rng(seed)
    truth, chans = {}, {}
    for c, (eye, size) in enumerate((("left", 1000.), ("right", 800.))):
        truth[eye] = size * np.exp((x @ bank.H[c]) / 100.0)
        y = truth[eye] * np.exp(rng.normal(0, 0.002, n))
        if gaps:
            for s in rng.integers(500, n - 500, int(8 * minutes)):
                y[s:s + rng.integers(8, 25)] = 0.0       # blinks
        chans[f"{eye}_pupil"] = y
    if gaps:
        chans["left_pupil"][6000:6800] = 0.0         # 8 s without signal
    d = pp.EyeData(sampling_rate=FS, name=name, inplace=inplace, **chans)
    return d.pupil.blinks_detect(strategies=["zero"], max_duration=None), truth


@pytest.fixture(scope="module")
def rec():
    return recording()


@pytest.fixture(scope="module")
def model(rec):
    d, _ = rec
    return pp.TimescaleModel(d, eyes="left", timescales=TS).fit()


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_preparation(rec):
    d, _ = rec
    m = pp.TimescaleModel(d, timescales=TS)
    assert m.eyes == m.units == ["left", "right"] and not m.binocular
    assert m.factor == 5 and m.fs == 20.0
    assert m.segments == [(0, 6000)] and len(m.time) == 6000
    assert m.time[0] == pytest.approx(d.tx[0] + 20.0)      # centre of 5 samples
    assert m.scale["left"] == pytest.approx(1000, rel=0.1)
    # log-% around 0, missing where the blinks and their margins are
    y = m.y["left"]
    assert abs(np.nanmedian(y)) < 1 and np.isnan(y[1200:1360]).all()
    usable = m._usable(d, "left")
    raw = d.data["left_pupil"]
    assert not usable[raw == 0].any()
    first = np.flatnonzero(raw == 0)[0]
    assert not usable[first - 10:first].any() and usable[first - 11]     # 100 ms before
    assert m.bank.timescales == (3000., 30000.)
    assert "not fitted" in repr(m)
    with pytest.raises(ValueError, match="not fitted"):
        m.summary()


def test_segments_from_excluded_periods_and_argument(rec):
    d, _ = rec
    dx = d.exclude_intervals([(60000, 90000)])
    m = pp.TimescaleModel(dx, eyes="left", timescales=TS)
    assert m.segments == [(0, 1200), (1200, 1200 + 4200)]
    assert m._spans == [(0, 6000), (9000, 30000)]
    assert m.time[1200] == pytest.approx(dx.tx[9000] + 20.0)
    # explicit segments split without excluding; data outside are not used
    m2 = pp.TimescaleModel(d, eyes="left", timescales=TS,
                           segments=[(0, 100 * pp.u.sec), ("100 s", "250.02 s")])
    assert m2._spans == [(0, 10000), (10000, 25002)]
    assert m2.segments == [(0, 2000), (2000, 2000 + 3001)]     # last block incomplete
    assert m2.n["left"][-1] <= 2
    with pytest.raises(ValueError, match="overlap"):
        pp.TimescaleModel(d, segments=[(0, 2000), (1000, 3000)])
    with pytest.raises(ValueError):
        pp.TimescaleModel(d, eyes="left", binocular=True)


def test_fit_and_summary(model):
    m = model
    p = m.params["left"]
    assert p["fast_sd"] == pytest.approx(8., rel=0.2)
    assert p["fast_ell"] == pytest.approx(700., rel=0.2)
    tab = m.summary(cutoff="3 s")
    assert list(tab.index) == ["left"]
    for col in ("fast_sd", "fast_ell", "noise_sd", "sd_3s", "sd_30s", "slow_sd", "n_obs",
                "loglik", "aic", "bic"):
        assert col in tab.columns
    assert tab.loc["left", "slow_sd"] == pytest.approx(
        np.hypot(p["sd_3s"], p["sd_30s"]))
    assert m.slow_sd("30 s")["left"] == pytest.approx(p["sd_30s"])
    assert set(m.loglik) == set(m.aic) == set(m.bic) == {"left"}
    assert m.aic["left"] < m.bic["left"]
    assert "fitted" in repr(m)


def test_components(model):
    parts = model.components(cutoff="10 s")["left"]
    assert set(parts) == {"mean", "fast", "3s", "30s", "slow", "detrended", "signal"}
    total = parts["mean"][0] + parts["fast"][0] + parts["3s"][0] + parts["30s"][0]
    assert np.allclose(total, parts["signal"][0])
    assert np.allclose(parts["mean"][0] + parts["30s"][0], parts["slow"][0])
    assert np.allclose(parts["fast"][0] + parts["3s"][0], parts["detrended"][0])
    assert parts["signal"][1][1300] > 3 * np.median(parts["signal"][1])    # long gap
    z = model.innovations()["left"]
    assert 0.8 < np.nanstd(z) < 1.2


def test_predict_at_original_samples(rec, model):
    d, truth = rec
    m = model
    est, lower, upper = m.predict("signal", interval=0.95)["left"]
    assert est.shape == (len(d),) and np.isfinite(est).all()
    t = truth["left"]
    assert np.sqrt(np.mean((est / t - 1)**2)) < 0.02
    # the long gap is bridged and the interval is honest
    gap = slice(6000, 6800)
    assert np.sqrt(np.mean((est[gap] / t[gap] - 1)**2)) < 0.12
    assert np.mean((lower < t) & (t < upper)) > 0.8
    assert np.mean(upper[gap] - lower[gap]) > 5 * np.median(upper - lower)
    # parts: slow + detrended = signal, on both scales
    slow = m.predict("slow", cutoff="10 s")["left"]
    rest = m.predict("detrended", cutoff="10 s")["left"]
    assert np.allclose(slow + rest, est)
    logs = {w: m.predict(w, scale="log%")["left"] for w in ("signal", "slow", "detrended")}
    assert np.allclose(logs["slow"] + logs["detrended"], logs["signal"])
    assert np.allclose(m.scale["left"] * np.exp(logs["signal"] / 100), est)
    assert np.std(np.diff(slow)) < 0.01 * np.std(np.diff(est))
    atom = m.predict(["30s", "mean"], scale="log%")["left"]
    assert np.allclose(atom, m.predict("slow", cutoff="30 s", scale="log%")["left"])
    # a range of samples only: same values there, NaN elsewhere
    part = m.predict("signal", _samples=(6100, 9003))["left"]
    assert np.allclose(part[6100:9003], est[6100:9003]) and np.isnan(part[:6000]).all()
    with pytest.raises(ValueError, match="intervals are only"):
        m.predict("detrended", interval=0.9)
    with pytest.raises(ValueError, match="scale"):
        m.predict(scale="mm")
    with pytest.raises(ValueError, match="no eye"):
        m.predict(eyes="right")


def test_interpolate(rec, model):
    d, truth = rec
    assert d.inplace is False
    n_hist = len(d.history)
    di = model.interpolate()
    assert di is not d and len(d.history) == n_hist
    assert "TimescaleModel.interpolate" in di.history[-1]["funcstring"]
    new, old = di.data["left_pupil"], d.data["left_pupil"]
    usable = model._usable(d, "left")
    assert np.array_equal(new[usable], old[usable])
    assert np.all(new > 0)
    assert np.allclose(new[~usable], model.predict("signal")["left"][~usable])
    assert np.sqrt(np.mean((new / truth["left"] - 1)**2)) < 0.03
    # usable under the default policy, not under the strict one
    assert not di.mask("left", "pupil").any()
    assert np.array_equal(di.mask("left", "pupil", "strict"), ~usable)
    assert np.array_equal(di.data["right_pupil"], d.data["right_pupil"])
    # other variable, exact blinks only, in place on a copy
    dc = d.copy()
    res = model.interpolate(dc, store_as="pupilinterp", margin=(0, 0), inplace=True)
    assert res is dc and np.array_equal(dc.data["left_pupil"], old)
    changed = dc.data["left_pupilinterp"] != old
    assert np.array_equal(changed, old == 0)
    with pytest.raises(ValueError, match="do not have the samples"):
        model.interpolate(d.crop(0, 10000, inplace=False))


def test_excluded_periods_are_not_interpolated(rec):
    d, _ = rec
    dx = d.exclude_intervals([(55000, 70000)])
    m = pp.TimescaleModel(dx, eyes="left", timescales=TS)
    m._set_fit("left", gm.BankFit(m.bank, m.bank.theta(
        fast_sd=8., fast_ell=700., noise_sd=0.2, sd_3s=3., sd_30s=5.), 0.0, 1, 2))
    est = m.predict("signal")["left"]
    assert np.isnan(est[5500:7000]).all() and np.isfinite(est[:5500]).all()
    di = m.interpolate()
    assert np.array_equal(di.data["left_pupil"][5500:7000], d.data["left_pupil"][5500:7000])
    assert (di.data["left_pupil"][:5500] > 0).all()
    # a segment without data of an eye is not reconstructed
    dn = d.copy()
    dn.data["left_pupil"] = np.where(np.arange(len(d)) < 5500, 0.0, d.data["left_pupil"])
    mn = pp.TimescaleModel(dn.exclude_intervals([(55000, 70000)]), timescales=TS)
    assert len(mn.segments) == 2 and np.isnan(mn.y["left"][:1100]).all()
    for eye in ("left", "right"):
        mn._set_fit(eye, m.fits["left"])
    est = mn.predict("signal")
    assert np.isnan(est["left"][:7000]).all() and np.isfinite(est["left"][7000:]).all()
    assert np.isfinite(est["right"][:5500]).all()


def test_store(rec, model):
    d, _ = rec
    ds = model.store("slow")
    assert "left_pupilslow" in ds.data and "left_pupilslow" not in d.data
    assert np.allclose(ds.data["left_pupilslow"], model.predict("slow")["left"])
    ds = model.store("signal", data=ds, inplace=True)
    ds = model.store("detrended", data=ds, cutoff="30 s", inplace=True)
    ds = model.store("fast", store_as="phasic", data=ds, inplace=True)
    assert {"left_pupilsmooth", "left_pupildetrended", "left_phasic"} <= set(ds.data.keys())
    assert "store('fast'" in ds.history[-1]["funcstring"]
    with pytest.raises(ValueError, match="store_as"):
        model.store(["3s"])


def test_checks(model):
    cal = model.forecast_calibration(horizons=("50 ms", "1 s", "20 s"))
    assert list(cal["horizon"]) == [0.05, 1.0, 20.0] and list(cal.columns) == \
        ["eye", "horizon", "rmse", "z_sd", "n"]
    assert np.all((cal["z_sd"] > 0.7) & (cal["z_sd"] < 1.3))
    chk = model.variogram_check(n_sim=10, rng=0)["left"]
    assert chk.outside < 0.5
    vg = model.variogram()["left"]
    assert vg.lags[0] == 50.0 and vg.lags[-1] <= 150000.0
    assert model.spectrum(window="50 s")["left"].nwindows >= 1
    cv = model.cross_validate(block="30 s", folds=2, refit=False)
    assert set(cv) == {"left"} and np.isfinite(cv["left"])


def test_keep_data_false(rec, model):
    d, _ = rec
    m = pp.TimescaleModel(d, eyes="left", timescales=TS, keep_data=False)
    m._set_fit("left", model.fits["left"])
    with pytest.raises(ValueError, match="keeps no reference"):
        m.interpolate()
    assert np.allclose(m.interpolate(d).data["left_pupil"],
                       model.interpolate().data["left_pupil"])


def test_binocular(rec):
    d, truth = rec
    m = pp.TimescaleModel(d, binocular=True, timescales=TS).fit(hessian=False, maxiter=15)
    assert m.units == ["both"] and m.eyes == ["left", "right"]
    p = m.params["both"]
    assert {"noise_sd_left", "noise_sd_right", "diff_sd", "diff_ell", "gain"} <= set(p)
    assert 0.5 < p["diff_sd"] < 6 and set(m.summary().index) == {"both"}
    # the parameters of the simulation (recovery is tested on arrays)
    truth_theta = m.bank.theta(fast_sd=8., fast_ell=700., noise_sd_left=0.2, noise_sd_right=0.2,
                               diff_sd=2., diff_ell=600., gain=1., sd_3s=3., sd_30s=5.)
    assert m.loglik["both"] > m.bank.loglik(truth_theta, m._Y("both"), m.fs, m.segments) - 300
    m._set_fit("both", gm.BankFit(m.bank, truth_theta, 0.0, 1))
    est = m.predict("signal")
    assert set(est) == {"left", "right"}
    # the left eye is missing for 8 s: the right eye carries it
    gap = slice(6000, 6800)
    err = np.sqrt(np.mean((est["left"][gap] / truth["left"][gap] - 1)**2))
    assert err < 0.03
    parts = m.components()
    assert "diff" in parts["left"]
    assert np.allclose(parts["left"]["diff"][0], -parts["right"]["diff"][0])
    assert np.allclose(parts["left"]["fast"][0], parts["right"]["fast"][0])
    di = m.interpolate()
    assert (di.data["left_pupil"] > 0).all() and (di.data["right_pupil"] > 0).all()
    assert len(m.forecast_calibration(horizons=("1 s",))) == 2
    m.plot.decomposition(eye="right")


def test_plots_follow_the_conventions(model):
    m = model
    for plot, kw in ((m.plot.variogram, {}), (m.plot.spectrum, dict(window="50 s")),
                     (m.plot.timescales, {}), (m.plot.slow, dict(cutoffs=("3 s", "30 s"))),
                     (m.plot.interpolation, dict(plot_range=(55, 75)))):
        fig, ax = plt.subplots()
        nfigs = len(plt.get_fignums())
        assert plot(ax=ax, **kw) is ax
        assert len(plt.get_fignums()) == nfigs and ax.has_data()
        fig, ax = plt.subplots()
        assert plot(**kw) is ax
        plt.close("all")
    for plot, kw in ((m.plot.decomposition, dict(plot_range=(1, 3))), (m.plot.diagnostics, {})):
        nfigs = len(plt.get_fignums())
        fig = plot(**kw)
        assert isinstance(fig, plt.Figure) and len(plt.get_fignums()) == nfigs + 1
        fig2, axs = plt.subplots(1, 3)
        assert plot(axes=axs, **kw) is fig2
        with pytest.raises(ValueError, match="axes needed"):
            plot(axes=list(axs) + [axs[0]], **kw)
        plt.close("all")
    m.plot.variogram(check=True, n_sim=5)
    assert "decomposition" in repr(m.plot)


def test_group():
    recs = {f"s{i}": recording(seed=10 + i, minutes=2, name=f"s{i}", gaps=False)[0]
            for i in range(4)}
    g = pp.TimescaleGroup(recs, eyes="left", timescales=("30 s",))
    assert list(g.models) == ["s0", "s1", "s2", "s3"] and "not fitted" in repr(g)
    with pytest.raises(ValueError, match="not fitted"):
        g.prior
    g.fit(n_iter=1, n_jobs=1)
    m = g.models["s0"]
    assert set(m.fits) == set(m.individual) == {"left"}
    assert m.fits["left"].prior is not None and m.fits["left"].bank is m.bank
    tab, ind = g.summary(), g.summary(individual=True)
    assert tab.shape[0] == 4 and tab.index.names == ["recording", "unit"]
    assert np.std(np.log(tab["sd_30s"] + 0.5)) <= np.std(np.log(ind["sd_30s"] + 0.5)) + 1e-6
    gs = g.group_summary()
    assert list(gs.index) == ["fast_sd", "fast_ell", "noise_sd", "sd_30s"]
    assert np.all(gs["low"] <= gs["typical"]) and np.all(gs["typical"] <= gs["high"])
    # the group as a prior for a new recording; models work as usual
    new = pp.TimescaleModel(recording(seed=99, minutes=2, gaps=False)[0], eyes="left",
                            timescales=("30 s",)).fit(prior=g.prior, hessian=False)
    assert new.fits["left"].prior.label.startswith("group")
    assert np.isfinite(m.predict("slow")["left"]).all()
    fig, axs = plt.subplots(1, 2)
    assert g.plot.params(ax=axs[0]) is axs[0] and g.plot.shrinkage(ax=axs[1]) is axs[1]
    # lists use the names of the objects; structures must agree
    g2 = pp.TimescaleGroup(list(g.models.values()))
    assert list(g2.models) == list(g.models)
    other = pp.TimescaleModel(recs["s0"], eyes="left", timescales=TS, name="other")
    with pytest.raises(ValueError, match="another structure"):
        pp.TimescaleGroup([m, other])
    with pytest.raises(ValueError, match="at least two"):
        pp.TimescaleGroup([m])


def test_group_from_files(tmp_path):
    files = []
    for i in range(2):
        d, _ = recording(seed=20 + i, minutes=1, name=f"f{i}", gaps=False)
        files.append(str(tmp_path / f"f{i}.pypil"))
        d.write_file(files[-1])
    g = pp.TimescaleGroup.from_files(files, preprocess=lambda d: d.exclude_intervals([(0, 1000)]),
                                     eyes="right", timescales=("30 s",))
    assert list(g.models) == ["f0", "f1"]
    m = g.models["f0"]
    assert m._data is None and m.eyes == ["right"] and m._spans == [(100, 6000)]
