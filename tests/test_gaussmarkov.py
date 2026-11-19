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
