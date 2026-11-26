"""Gauss-Markov state-space machinery against dense Gaussian computations."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from pypillometry.signal import gaussmarkov as gm
from pypillometry.signal import semivariogram
from pypillometry.signal.gaussmarkov.components import Component

FS = 250.0
DT = 1000.0 / FS


def _model():
    return gm.GaussMarkovModel([gm.OU(1.0, 30.0), gm.Matern(2.0, 200.0, nu=2.5)], noise=0.3)


def _dense_cov(model, T):
    t = np.arange(T) * DT
    return model.cov(t[:, None] - t[None, :])


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, 3.5])
def test_matern_state_space_matches_closed_form(nu):
    m = gm.Matern(1.5, 80.0, nu=nu)
    tau = np.array([0.0, 5.0, 40.0, 200.0])
    np.testing.assert_allclose(Component.cov(m, tau), m.cov(tau), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(m.ss().Pinf[0, 0], 1.5**2)


@pytest.mark.parametrize("comp", [gm.OU(1.0, 30.0), gm.Matern(2.0, 200.0, nu=2.5)])
def test_discretisation_keeps_stationary_covariance(comp):
    A, Q = comp.discretise(DT)
    P = comp.ss().Pinf
    np.testing.assert_allclose(A @ P @ A.T + Q, P, rtol=1e-8, atol=1e-12)


def test_parameters_roundtrip():
    model = _model().fix({"noise": 0.5})
    assert model.free_names == ["ou0.sigma", "ou0.tau", "matern1.sigma", "matern1.ell"]
    assert model.noise == 0.5
    new = model.from_theta(model.theta + np.log(2))
    assert new.params["ou0.tau"] == pytest.approx(60.0)
    assert new.params["noise"] == 0.5
    assert new.components["matern1"].nu == 2.5
    assert model.unfix(["noise"]).free_names[-1] == "noise"
    with pytest.raises(ValueError):
        model.with_params({"ou0.ell": 1.0})


@pytest.mark.parametrize("gaps", [False, True])
def test_loglik_matches_dense_gaussian(gaps):
    model = _model()
    T = 150
    y = gm.simulate(model, T, fs=FS, rng=3)
    C = _dense_cov(model, T)
    valid = np.ones(T, bool)
    if gaps:
        valid[20:45] = False
        valid[100] = False
        y[~valid] = np.nan
    expected = multivariate_normal(np.zeros(valid.sum()), C[np.ix_(valid, valid)]).logpdf(y[valid])
    ssm = model.to_lgssm(FS)
    fres = gm.kalman_filter(ssm, y)
    assert fres.loglik == pytest.approx(expected, rel=1e-8)
    assert gm.kalman_loglik(ssm, y) == pytest.approx(expected, rel=1e-8)
    assert fres.loglik_t.sum() == pytest.approx(expected, rel=1e-8)
    assert np.isnan(fres.innov[~valid]).all()
    assert gm.loglik_signal(model, y, FS, mean_tol=None) == pytest.approx(expected, rel=1e-8)


def test_smoother_matches_gp_posterior():
    model = _model()
    T = 120
    y = gm.simulate(model, T, fs=FS, rng=4)
    y[30:60] = np.nan
    v = np.isfinite(y)
    ssm = model.to_lgssm(FS)
    sres = gm.rts_smoother(ssm, gm.kalman_filter(ssm, y))
    f_mean = sres.m_smooth @ ssm.H[0]
    f_var = np.einsum("i,tij,j->t", ssm.H[0], sres.P_smooth, ssm.H[0])

    Kf = _dense_cov(model, T) - model.noise**2 * np.eye(T)
    Kyy = Kf[np.ix_(v, v)] + model.noise**2 * np.eye(v.sum())
    Kfy = Kf[:, v]
    gp_mean = Kfy @ np.linalg.solve(Kyy, y[v])
    gp_cov = Kf - Kfy @ np.linalg.solve(Kyy, Kfy.T)
    np.testing.assert_allclose(f_mean, gp_mean, atol=1e-7)
    np.testing.assert_allclose(f_var, np.diag(gp_cov), atol=1e-7)
    np.testing.assert_allclose(sres.signal, gp_mean, atol=1e-7)
    np.testing.assert_allclose(sres.signal_var, np.diag(gp_cov), atol=1e-7)

    parts = gm.component_states(model, sres)
    np.testing.assert_allclose(parts["ou0"][0] + parts["matern1"][0], f_mean)


def test_lag1_covariance_matches_gp_posterior():
    model = gm.GaussMarkovModel([gm.OU(1.0, 30.0)], noise=0.5)
    T = 60
    y = gm.simulate(model, T, fs=FS, rng=5)
    ssm = model.to_lgssm(FS)
    sres = gm.rts_smoother(ssm, gm.kalman_filter(ssm, y))
    Kf = _dense_cov(model, T) - model.noise**2 * np.eye(T)
    post = Kf - Kf @ np.linalg.solve(Kf + model.noise**2 * np.eye(T), Kf)
    np.testing.assert_allclose(sres.P_lag1[1:, 0, 0], np.diag(post, -1), atol=1e-9)


def test_multichannel_partial_missing():
    # one OU observed by two channels with different noise
    ou = gm.OU(1.0, 30.0)
    A, Q = ou.discretise(DT)
    R = np.diag([0.2, 0.5])**2
    ssm = gm.LGSSM(A=A, Q=Q, H=np.ones((2, 1)), R=R, m0=np.zeros(1), P0=ou.ss().Pinf)
    T = 80
    y = gm.simulate(ssm, T, rng=6)
    y[10:20, 0] = np.nan
    y[15:30, 1] = np.nan
    t = np.arange(T) * DT
    Kf = ou.cov(t[:, None] - t[None, :])
    C = np.kron(Kf, np.ones((2, 2))) + np.kron(np.eye(T), R)
    v = np.isfinite(y.ravel())
    expected = multivariate_normal(np.zeros(v.sum()), C[np.ix_(v, v)]).logpdf(y.ravel()[v])
    fres = gm.kalman_filter(ssm, y)
    assert fres.loglik == pytest.approx(expected, rel=1e-8)
    assert fres.std_innov.shape == (T, 2)
    assert np.all(fres.gain[10:15, :, 0] == 0)
    assert fres.signal.shape == fres.signal_var.shape == (T, 2)
    np.testing.assert_allclose(fres.signal[:, 0], fres.m_filt[:, 0])


def test_simulated_variogram_matches_model():
    model = gm.GaussMarkovModel([gm.OU(1.0, 20.0), gm.Matern(0.5, 100.0, nu=1.5)], noise=0.3)
    y = gm.simulate(model, 200_000, fs=1000.0, rng=7)
    vg = semivariogram(y, fs=1000.0, max_lag=100)
    np.testing.assert_allclose(vg.gamma, model.variogram(vg.lags), rtol=0.1)


def test_loglik_variogram():
    model = _model()
    vg = semivariogram(gm.simulate(model, 20_000, fs=FS, rng=8), fs=FS, max_lag=400)
    ll = gm.loglik_variogram(model, vg)
    wrong = model.with_params({"ou0.tau": 300.0})
    assert ll > gm.loglik_variogram(wrong, vg)
    assert gm.loglik_variogram(model, vg, method="wls") > \
        gm.loglik_variogram(wrong, vg, method="wls")
    with pytest.raises(ValueError):
        gm.loglik_variogram(model, vg, method="foo")


def test_fit_recovers_parameters():
    true = gm.GaussMarkovModel([gm.OU(1.0, 50.0)], noise=0.3)
    y = gm.simulate(true, 20_000, fs=FS, rng=9)
    y[5000:5500] = np.nan
    y -= np.nanmean(y)
    start = gm.GaussMarkovModel([gm.OU(0.5, 150.0)], noise=0.6)

    res = gm.fit(start, signal=y, fs=FS)
    for k, v in true.params.items():
        assert res.model.params[k] == pytest.approx(v, rel=0.15), k
    assert res.n_obs == 19_500 and res.n_params == 3
    assert res.aic == pytest.approx(-2 * res.loglik + 6)
    assert res.bic == pytest.approx(-2 * res.loglik + 3 * np.log(19_500))

    vg = semivariogram(y, fs=FS, max_lag=500)
    res = gm.fit(start, vg=vg, which="variogram", neff_scale=1e-3)
    for k, v in true.params.items():
        assert res.model.params[k] == pytest.approx(v, rel=0.25), k
    assert res.n_obs is None
    with pytest.raises(ValueError):
        res.aic


def test_loglik_signal_requires_zero_mean():
    model = gm.GaussMarkovModel([gm.OU(1.0, 30.0)], noise=0.3)
    y = gm.simulate(model, 2000, fs=FS, rng=10)
    y -= y.mean()
    gm.loglik_signal(model, y, FS)
    with pytest.raises(ValueError, match="zero-mean"):
        gm.loglik_signal(model, y + 5.0, FS)
    with pytest.raises(ValueError, match="zero-mean"):
        gm.fit(model, signal=y + 5.0, fs=FS)
    gm.loglik_signal(model, y + 5.0, FS, mean_tol=None)


def test_heavy_computation_warns(monkeypatch):
    from pypillometry.signal.gaussmarkov import kalman
    messages = []
    monkeypatch.setattr(kalman.logger, "warning", messages.append)
    monkeypatch.setattr(kalman, "_MAX_SEC", 0.0)
    monkeypatch.setattr(kalman, "_WARNED", set())
    ssm = _model().to_lgssm(FS)
    gm.kalman_loglik(ssm, np.zeros(10))
    gm.kalman_loglik(ssm, np.zeros(10))
    assert len(messages) == 1 and "kalman_loglik" in messages[0]


# -- diffuse mean, segments, noise variance per sample ------------------------

def _mean_model():
    comps = {"f": gm.Matern(3., 800.), "s": gm.Matern(2., 20000.)}
    plain = gm.GaussMarkovModel(comps, noise=0.3)
    return gm.GaussMarkovModel({"mean": gm.Constant(), **comps}, noise=0.3), plain


def _gappy(plain, n=3000, fs=20):
    y = gm.simulate(plain, n, fs=fs, rng=1)
    y[500:700] = np.nan
    y[1900:2500] = np.nan
    y[np.random.default_rng(0).random(n) < 0.1] = np.nan
    return y


def test_constant_component():
    c = gm.Constant()
    assert c.dim == 1 and c.param_names == ()
    assert np.all(c.variogram([10., 100.]) == 0)
    model, plain = _mean_model()
    assert model.n_diffuse == 1 and plain.n_diffuse == 0
    assert model.free_names == plain.free_names
    assert np.allclose(model.variogram([50., 500.]), plain.variogram([50., 500.]))


@pytest.mark.parametrize("comp", [gm.OU(1., 50.), gm.Matern(2., 300., nu=1.5),
                                  gm.Matern(1., 100., nu=2.5)])
def test_psd_integrates_to_variance(comp):
    f = np.linspace(0, 400, 80001)
    assert np.isclose(np.trapezoid(comp.psd(f), f), comp.sigma**2, rtol=2e-2)
    model = gm.GaussMarkovModel([comp, gm.Constant()])
    assert np.allclose(model.psd(f[:5]), comp.psd(f[:5]))


def test_diffuse_mean_gives_restricted_likelihood():
    """invariant to the level, and equal to the dense REML likelihood"""
    from scipy.linalg import cho_factor, cho_solve
    model, plain = _mean_model()
    y = _gappy(plain)[:2200]
    l0 = gm.loglik_signal(model, y, 20)
    assert abs(gm.loglik_signal(model, y + 1000, 20) - l0) < 1e-2
    v = np.isfinite(y)
    t = np.arange(len(y))[v] * 50.
    for pars in ({}, {"f.sigma": 4., "s.ell": 9000., "noise": 0.5}):
        c = cho_factor(plain.with_params(pars).cov(t[:, None] - t[None, :]))
        X = np.ones(v.sum())
        xkx = X @ cho_solve(c, X)
        r = y[v] - X * (X @ cho_solve(c, y[v])) / xkx
        reml = -0.5 * (2 * np.sum(np.log(np.diag(c[0]))) + np.log(xkx)
                       + r @ cho_solve(c, r) + (v.sum() - 1) * np.log(2 * np.pi))
        assert np.isclose(gm.loglik_signal(model.with_params(pars), y, 20), reml,
                          atol=1e-4)


def test_segments_are_independent():
    model, plain = _mean_model()
    y = _gappy(plain)
    segs = [(0, 1500), (1700, 2900)]
    parts = sum(gm.loglik_signal(model, y[a:b], 20) for a, b in segs)
    assert np.isclose(gm.loglik_signal(model, y, 20, segments=segs), parts)
    ssm = model.to_lgssm(20)
    fres = gm.kalman_filter(ssm, y, n_diffuse=1, segments=segs)
    assert np.isclose(fres.loglik, parts)
    assert not fres.valid[1500:1700].any() and not fres.valid[2900:].any()
    sres = gm.rts_smoother(ssm, fres)
    for a, b in segs:   # same as smoothing the segment alone
        alone = gm.rts_smoother(ssm, gm.kalman_filter(ssm, y[a:b], n_diffuse=1))
        assert np.allclose(sres.m_smooth[a:b], alone.m_smooth)
        assert np.allclose(sres.P_smooth[a:b], alone.P_smooth)
    # the mean is a constant within a segment and differs between them
    mu = sres.m_smooth[:, 0]
    assert np.ptp(mu[:1500]) < 1e-6 and np.ptp(mu[1700:2900]) < 1e-6
    with pytest.raises(ValueError, match="segments"):
        gm.kalman_loglik(ssm, y, segments=[(0, 100), (50, 200)])


def test_obs_var_per_sample():
    model, plain = _mean_model()
    y = _gappy(plain)
    ssm, pssm = model.to_lgssm(20), plain.to_lgssm(20)
    const = np.full(len(y), plain.noise**2)
    # constant obs_var is R; sequential and general kernel agree
    assert np.isclose(gm.kalman_loglik(pssm, y, obs_var=const), gm.kalman_loglik(pssm, y))
    f1, f2 = gm.kalman_filter(pssm, y, obs_var=const), gm.kalman_filter(pssm, y)
    assert np.allclose(f1.m_filt, f2.m_filt) and np.allclose(f1.P_filt, f2.P_filt)
    assert np.allclose(f1.std_innov, f2.std_innov, equal_nan=True)
    # a sample with a huge noise variance is as good as missing
    big = const.copy()
    big[100] = 1e12
    y2 = y.copy()
    y2[100] = np.nan
    m1 = gm.kalman_filter(ssm, y, obs_var=big, n_diffuse=1).m_filt
    assert np.allclose(m1, gm.kalman_filter(ssm, y2, n_diffuse=1).m_filt, atol=1e-6)
    with pytest.raises(ValueError, match="obs_var"):
        gm.kalman_loglik(pssm, y, obs_var=const[:10])


def test_sequential_kernel_two_channels():
    """two channels with independent noise: same as the joint update"""
    comp = gm.Matern(2., 1000.)
    s = comp.ss()
    A, Q = comp.discretise(50.)
    ssm = gm.LGSSM(A=A, Q=Q, H=np.vstack([s.H, s.H]), R=np.diag([0.1, 0.4]),
                   m0=np.zeros(2), P0=s.Pinf)
    y = gm.simulate(ssm, 500, rng=3)
    y[100:150, 0] = np.nan
    y[130:200, 1] = np.nan
    r = np.tile([0.1, 0.4], (500, 1))
    assert np.isclose(gm.kalman_loglik(ssm, y, obs_var=r), gm.kalman_loglik(ssm, y))
    f1, f2 = gm.kalman_filter(ssm, y, obs_var=r), gm.kalman_filter(ssm, y)
    assert np.allclose(f1.m_filt, f2.m_filt) and np.allclose(f1.P_filt, f2.P_filt)
    ssm.R = np.array([[0.1, 0.05], [0.05, 0.4]])
    with pytest.raises(ValueError, match="diagonal"):
        gm.kalman_loglik(ssm, y, n_diffuse=1)
