"""Timescale-bank model on arrays: pp.signal.gaussmarkov (bank, hierarchical, check)."""
import pickle

import numpy as np
import pytest

from pypillometry.signal import gaussmarkov as gm
from pypillometry.signal import block_mean, semivariogram

FS = 20.0


@pytest.fixture(scope="module")
def bank():
    return gm.TimescaleBank(timescales=(3000., 30000.))


@pytest.fixture(scope="module")
def sim(bank):
    """6 min at 20 Hz with gaps, a level of 5 and its states"""
    theta = bank.theta(fast_sd=8., fast_ell=700., noise_sd=0.2, sd_3s=3., sd_30s=5.)
    y, x = bank.simulate(theta, 7200, FS, rng=1, return_states=True)
    y = y + 5.0
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 7200, 150):
        y[s:s + rng.integers(2, 12)] = np.nan
    y[3000:3200] = np.nan
    return theta, y, x


@pytest.fixture(scope="module")
def fitted(bank, sim):
    theta, y, x = sim
    return gm.fit_bank(bank, y, FS, prior=gm.reference_prior(bank))


def test_structure_and_parameters(bank):
    assert bank.names == ["log_fast_sd", "log_fast_ell", "log_noise_sd", "u_3s", "u_30s"]
    assert bank.component_names == ["mean", "fast", "3s", "30s"]
    assert bank.dim == 7 and bank.K == 2 and bank.n_channels == 1
    pars = dict(fast_sd=8., fast_ell=700., noise_sd=0.2, sd_3s=3., sd_30s=0.)
    theta = bank.theta(**pars)
    assert np.isclose(theta[-1], np.log(bank.v0))       # no variance: finite u
    assert bank.params(theta) == pytest.approx(pars)
    assert bank.slow_sd(theta, cutoff=3000.) == pytest.approx(3.)
    assert bank.slow_sd(theta, cutoff=10000.) == 0.
    lo, hi = np.array(bank.bounds()).T
    assert np.all(lo <= theta) and np.all(theta <= hi)
    # below the bound the variance stays 0
    theta[-1] -= 1
    assert bank.atom_var(theta)[-1] == 0
    with pytest.raises(ValueError):
        bank.theta(fast_sd=1.)
    with pytest.raises(ValueError):
        gm.TimescaleBank(timescales=(30000., 3000.))
    assert pickle.loads(pickle.dumps(bank)) == bank


def test_select_weights(bank):
    w = lambda *a, **k: bank.select(*a, **k)
    assert np.array_equal(w("signal"), bank.H[0])
    assert np.array_equal(w("slow", cutoff=10000.) + w("detrended", cutoff=10000.), w("signal"))
    assert np.array_equal(w("fast") + w("3s"), w("detrended", cutoff=10000.))
    assert np.array_equal(w("slow", cutoff=3000.), w(["mean", "3s", "30s"]))
    assert w("30s").sum() == 1 and w("30s")[bank.slices["30s"].start] == 1
    with pytest.raises(ValueError, match="unknown components"):
        w("nope")


def test_model_matches_gauss_markov_model(bank, sim):
    """same likelihood, variogram and spectrum as the generic model"""
    theta, y, _ = sim
    p = bank.params(theta)
    model = gm.GaussMarkovModel(
        {"mean": gm.Constant(), "fast": gm.Matern(p["fast_sd"], p["fast_ell"]),
         "a": gm.Matern(p["sd_3s"], 3000.), "b": gm.Matern(p["sd_30s"], 30000.)},
        noise=p["noise_sd"])
    assert np.isclose(bank.loglik(theta, y, FS), gm.loglik_signal(model, y, FS))
    lags = np.array([50., 1000., 50000.])
    assert np.allclose(bank.variogram(theta, lags), model.variogram(lags))
    f = np.array([0.01, 0.5, 5.])
    assert np.allclose(bank.psd(theta, f), model.psd(f))
    assert np.allclose(bank.psd(theta, f, fs=FS), model.psd(f) + 0.04 / 10)
    # independent of the level, additive over segments
    assert np.isclose(bank.loglik(theta, y + 100, FS), bank.loglik(theta, y, FS), atol=1e-3)
    segs = [(0, 3000), (3200, 7200)]
    parts = sum(bank.loglik(theta, y[a:b], FS) for a, b in segs)
    assert np.isclose(bank.loglik(theta, y, FS, segments=segs), parts)


def test_start_from_variogram(bank, sim):
    theta, y, _ = sim
    vg = semivariogram(y, FS, max_lag=180000)
    start = bank.start(vg)
    lo, hi = np.array(bank.bounds()).T
    assert np.all(lo <= start) and np.all(start <= hi)
    p = bank.params(start)
    assert 4 < p["fast_sd"] < 12 and 300 <= p["fast_ell"] <= 1500
    # better than a far-off guess
    off = bank.theta(fast_sd=1., fast_ell=200., noise_sd=1., sd_3s=1., sd_30s=1.)
    assert bank.loglik(start, y, FS) > bank.loglik(off, y, FS)


def test_fit_recovers_parameters(bank, sim, fitted):
    theta, y, x = sim
    fit = fitted
    assert fit.success and fit.n_obs == np.isfinite(y).sum()
    assert fit.loglik >= bank.loglik(theta, y, FS) - 3
    p = fit.params
    assert p["fast_sd"] == pytest.approx(8., rel=0.1)
    assert p["fast_ell"] == pytest.approx(700., rel=0.1)
    assert p["noise_sd"] == pytest.approx(0.2, rel=0.25)
    assert np.all(np.abs(fit.theta - theta) < 3.5 * fit.se)
    # the well-determined parameters have small standard errors
    assert np.all(fit.se[:3] < 0.1) and np.all(fit.se[3:] > 0.1)
    assert fit.aic == pytest.approx(-2 * fit.loglik + 2 * 5)
    assert fit.bic == pytest.approx(-2 * fit.loglik + 5 * np.log(fit.n_obs - 1))
    assert fit.logpost < fit.loglik


def test_smoother_recovers_signal_and_slow_part(bank, sim, fitted):
    theta, y, x = sim
    fres, sres = bank.smooth(fitted.theta, y, FS)
    truth = x @ bank.H[0] + 5.0
    w = bank.select("signal")
    est = sres.m_smooth @ w
    sd = np.sqrt(np.einsum("i,tij,j->t", w, sres.P_smooth, w))
    assert np.sqrt(np.mean((est - truth)**2)) < 1.5
    z = (est - truth) / sd
    assert 0.7 < np.std(z) < 1.3
    assert sd[3100] > 5 * np.median(sd)         # wide in the long gap
    ws = bank.select("slow", cutoff=10000.)
    slow_true = x @ bank.select("30s") + 5.0
    assert np.corrcoef(sres.m_smooth @ ws, slow_true)[0, 1] > 0.8
    # innovations are standardised
    zi = fres.std_innov[1:, 0]
    assert 0.9 < np.nanstd(zi) < 1.1


def test_prior_pulls_estimates(bank, sim, fitted):
    theta, y, _ = sim
    names = bank.names
    target = fitted.theta.copy()
    target[-1] += 2.0          # u of the 30-s atom
    tight = gm.GaussianPrior(target, np.full(5, 0.05), names)
    fit = gm.fit_bank(bank, y, FS, prior=tight, theta0=fitted.theta, hessian=False)
    assert abs(fit.theta[-1] - target[-1]) < abs(fitted.theta[-1] - target[-1]) / 2
    assert fit.loglik < fitted.loglik
    ml = gm.fit_bank(bank, y, FS, prior=None, theta0=fitted.theta, hessian=False)
    assert ml.loglik >= fitted.loglik - 1e-6 and ml.prior is None
    with pytest.raises(ValueError, match="prior is for"):
        gm.fit_bank(bank, y, FS, prior=gm.general_prior())
    with pytest.raises(TypeError):
        gm.fit_bank(bank, y, FS, prior=3)


def test_general_prior():
    bank = gm.TimescaleBank()
    medium = gm.general_prior(bank)
    assert medium.names == bank.names and medium.cov.shape == (8, 8)
    assert np.allclose(gm.general_prior(bank, "weak").sd, 2 * medium.sd)
    assert np.allclose(gm.general_prior(bank, "strong").sd, 0.5 * medium.sd)
    assert np.allclose(gm.general_prior(bank, 3.0).mean, medium.mean)
    # other time scales: interpolated over the log time scale
    other = gm.general_prior(gm.TimescaleBank(timescales=(3000., 17320.5, 1e6)))
    assert other.mean[3] == medium.mean[3]
    assert np.isclose(other.mean[4], 0.5 * (medium.mean[4] + medium.mean[5]), atol=1e-3)
    assert other.mean[5] == medium.mean[7]
    # two channels: the noise prior for both, and one for the difference
    two = gm.general_prior(gm.TimescaleBank(channels=("left", "right")))
    assert len(two.mean) == 12 and two.mean[2] == two.mean[3] == medium.mean[2]
    assert two.names[6] == "log_gain" and two.mean[6] == 0
    with pytest.raises(ValueError):
        gm.general_prior(bank, "huge")
    assert "log_fast_sd" in repr(medium)


def test_upsample_is_exact(bank, sim):
    """the bridge between the samples equals the dense GP posterior"""
    theta, _, _ = sim
    fine, x = bank.simulate(theta, 25 * 300, 500, rng=2, return_states=True)
    y, _ = block_mean(fine, 25)
    y[100:105] = np.nan
    y[200:230] = np.nan
    fres, sres = bank.smooth(theta, y, FS)
    mean, var = bank.upsample(theta, FS, sres, 25, bank.select("signal"))
    assert mean.shape == var.shape == (7500,)
    v = np.isfinite(y)
    tc = (np.arange(300) * 25 + 12) * 2.0       # centres of the blocks (ms)
    tf = np.arange(7500) * 2.0
    k = lambda d: sum(c.cov(d) for _, c, _ in bank._components(theta))
    Kc = k(tc[v][:, None] - tc[v][None, :]) + 0.04 * np.eye(v.sum()) + bank.kappa
    Kfc = k(tf[:, None] - tc[v][None, :]) + bank.kappa
    dense = Kfc @ np.linalg.solve(Kc, y[v])
    dvar = k(0.) + bank.kappa - np.einsum("ij,ji->i", Kfc, np.linalg.solve(Kc, Kfc.T))
    assert np.allclose(mean, dense, atol=1e-4)
    assert np.allclose(var, dvar, atol=1e-4)
    # at the centres it is the smoothed value; several parts at once
    both, _ = bank.upsample(theta, FS, sres, 25, [bank.select("signal"), bank.select("slow")])
    assert np.allclose(both[0][12::25], sres.m_smooth @ bank.select("signal"))
    assert np.allclose(both[0], mean)
    # segments: NaN outside, each one on its own
    segs = [(0, 150), (160, 300)]
    fres, sres = bank.smooth(theta, y, FS, segments=segs)
    m2, _ = bank.upsample(theta, FS, sres, 25, bank.select("signal"), segments=segs)
    assert np.isnan(m2[150 * 25:160 * 25]).all() and np.isfinite(m2[:150 * 25]).all()
    fa, sa = bank.smooth(theta, y[:150], FS)
    assert np.allclose(m2[:150 * 25], bank.upsample(theta, FS, sa, 25, bank.select("signal"))[0])


def test_upsample_with_atom_without_variance(bank, sim):
    theta, y, _ = sim
    th = theta.copy()
    th[-1] = np.log(bank.v0)
    fres, sres = bank.smooth(th, y[:400], FS)
    mean, var = bank.upsample(th, FS, sres, 5, bank.select("signal"))
    assert np.isfinite(mean).all() and np.all(var >= 0)


def test_two_channels():
    bank = gm.TimescaleBank(timescales=(3000., 30000.), channels=("left", "right"))
    assert bank.names[2:7] == ["log_noise_sd_left", "log_noise_sd_right",
                               "log_diff_sd", "log_diff_ell", "log_gain"]
    assert bank.component_names == ["mean_left", "mean_right", "fast", "diff", "3s", "30s"]
    pars = dict(fast_sd=8., fast_ell=700., noise_sd_left=0.2, noise_sd_right=0.4,
                diff_sd=2.5, diff_ell=600., sd_3s=3., sd_30s=5.)
    theta = bank.theta(gain=1., **pars)
    Y, X = bank.simulate(theta, 6000, FS, rng=1, return_states=True)
    Y = Y + [3., -4.]
    # the channels differ by the difference component and the noise
    d = Y[:, 0] - Y[:, 1] - 7
    assert np.var(d) == pytest.approx(2.5**2 + 0.2**2 + 0.4**2, rel=0.3)
    assert bank.variogram(theta, [1e7], channel=0)[0] == pytest.approx(
        64 + 9 + 25 + 2.5**2 / 4 + 0.04)
    assert bank.variogram(theta, [1e7], channel=1)[0] == pytest.approx(
        64 + 9 + 25 + 2.5**2 / 4 + 0.16)
    # a gain between the channels: the common part is larger in the second
    gained = bank.theta(gain=1.44, **pars)
    assert np.allclose(bank.gains(gained), [1 / 1.2, 1.2]) and np.all(bank.gains() == 1)
    assert bank.variogram(gained, [1e7], channel=1)[0] == pytest.approx(
        1.44 * (64 + 9 + 25) + 2.5**2 / 4 + 0.16)
    assert bank.psd(gained, [0.1], channel=0)[0] < bank.psd(theta, [0.1], channel=0)[0]
    Hg = bank.obs_matrix(gained)
    assert np.allclose(Hg[1, [2, 6, 8]], 1.2) and np.allclose(Hg[:, 4], [0.5, -0.5])
    assert np.array_equal(bank.select("signal", channel=1, theta=gained), Hg[1])
    assert np.allclose(bank.select("fast", channel=0, theta=gained)[2:4], [1 / 1.2, 0])
    Yg = bank.simulate(gained, 3000, FS, rng=1) + [3., -4.]
    fg = gm.fit_bank(bank, Yg, FS, prior=gm.reference_prior(bank), hessian=False)
    assert fg.params["gain"] == pytest.approx(1.44, rel=0.1)
    rng = np.random.default_rng(0)
    Y[rng.random(6000) < 0.1, 0] = np.nan
    Y[2000:2300, 1] = np.nan
    assert np.isclose(bank.loglik(theta, Y + [50., -20.], FS), bank.loglik(theta, Y, FS),
                      atol=1e-3)
    fit = gm.fit_bank(bank, Y, FS, prior=gm.reference_prior(bank), hessian=False)
    p = fit.params
    assert fit.loglik >= bank.loglik(theta, Y, FS) - 3
    assert p["noise_sd_right"] > p["noise_sd_left"]
    assert p["diff_sd"] == pytest.approx(2.5, rel=0.2)
    # an eye is reconstructed from the other where it is missing
    fres, sres = bank.smooth(fit.theta, Y, FS)
    est = sres.m_smooth @ bank.select("signal", channel=1, theta=fit.theta)
    truth = X @ bank.H[1] - 4.0
    gap = slice(2000, 2300)
    assert np.sqrt(np.mean((est - truth)[gap]**2)) < 3.0     # sd of the signal: 10
    assert np.allclose(bank.select("slow", channel=0)[:2], [1, 0])
    assert np.allclose(bank.select("mean", channel=1)[:2], [0, 1])


def test_forecast_errors_are_calibrated(bank, sim):
    theta, y, _ = sim
    E, Z = bank.forecast_errors(theta, y, FS, [1, 20, 400], step=5)
    assert E.shape == Z.shape == (1440, 3, 1)
    sd = np.nanstd(Z[:, :, 0], axis=0)
    assert np.all((0.8 < sd) & (sd < 1.2))
    rmse = np.sqrt(np.nanmean(E[:, :, 0]**2, axis=0))
    assert rmse[0] < rmse[1] < rmse[2]
    # a model without slow components is too confident over 20 s
    wrong = bank.theta(fast_sd=8., fast_ell=700., noise_sd=0.2)
    _, Zw = bank.forecast_errors(wrong, y, FS, [400], step=5)
    assert np.nanstd(Zw) > 1.2
    with pytest.raises(ValueError):
        bank.forecast_errors(theta, y, FS, [0])


def test_blocked_cv_prefers_the_true_model(bank, sim):
    theta, y, _ = sim
    good = gm.blocked_cv(bank, theta, y, FS, block="20 s", refit=False)
    wrong = bank.theta(fast_sd=8., fast_ell=700., noise_sd=0.2)
    bad = gm.blocked_cv(bank, wrong, y, FS, block="20 s", refit=False)
    assert good > bad + 20
    refit = gm.blocked_cv(bank, theta, y, FS, block="60 s", folds=2, refit=True, maxiter=3,
                          prior=gm.reference_prior(bank))
    assert np.isfinite(refit)


def test_variogram_check(bank, sim):
    theta, y, _ = sim
    chk = gm.variogram_check(bank, theta, y, FS, n_sim=20, rng=0)
    assert len(chk.lags) == len(chk.gamma) == len(chk.lower) == len(chk.upper)
    assert np.all(chk.lower <= chk.upper)
    assert chk.outside <= 0.4
    wrong = bank.theta(fast_sd=2., fast_ell=700., noise_sd=0.2)
    assert gm.variogram_check(bank, wrong, y, FS, n_sim=20, rng=0).outside > 0.8


def test_group_fit_shrinks_towards_the_group():
    bank = gm.TimescaleBank(timescales=(30000.,))
    rng = np.random.default_rng(5)
    mu = bank.theta(fast_sd=8., fast_ell=700., noise_sd=0.2, sd_30s=4.)
    sd = np.array([0.3, 0.1, 0.2, 0.4])
    thetas = mu + sd * rng.standard_normal((6, 4))
    data = [(bank.simulate(th, 2000, FS, rng=i), FS, None) for i, th in enumerate(thetas)]
    g = gm.fit_group(bank, data, n_iter=2, n_jobs=1)
    assert len(g.fits) == len(g.individual) == 6 and len(g.history) >= 2
    assert g.mu.shape == (4,) and g.Sigma.shape == (4, 4)
    assert np.allclose(g.Sigma, np.diag(np.diag(g.Sigma)))
    assert np.all(np.abs(g.mu[:3] - mu[:3]) < 0.4)
    ind = np.array([f.theta for f in g.individual])
    # the poorly determined atom is shrunk, the fast component is not
    assert np.std(g.thetas[:, 3]) < np.std(ind[:, 3])
    assert np.allclose(g.thetas[:, 0], ind[:, 0], atol=0.1)
    # shrunk estimates are not further from the truth
    err = lambda th: np.mean((th[:, 3] - thetas[:, 3])**2)
    assert err(g.thetas) <= err(ind) * 1.1
    prior = g.prior
    assert isinstance(prior, gm.GaussianPrior) and prior.names == bank.names
    assert all(f.prior is not None for f in g.fits)
    # reuse of the individual fits, full covariance
    g2 = gm.fit_group(bank, data, n_iter=1, n_jobs=1, cov="full", individual=g.individual)
    assert not np.allclose(g2.Sigma, np.diag(np.diag(g2.Sigma)))
    with pytest.raises(ValueError):
        gm.fit_group(bank, data[:1])
    with pytest.raises(ValueError):
        gm.fit_group(bank, data, cov="banana")
