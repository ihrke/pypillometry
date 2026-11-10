"""Semivariogram of signals with missing samples (pp.signal.variogram)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pypillometry.signal import semivariogram, plot_variogram


@pytest.fixture
def gappy():
    """Random walk plus noise, 20 s at 500 Hz, with masked gaps (set to 0)."""
    rng = np.random.default_rng(3)
    n = 10000
    x = 4000 + np.cumsum(rng.normal(0, 1, n)) + rng.normal(0, 3, n)
    mask = np.zeros(n, dtype=bool)
    for start in rng.integers(0, n - 300, 15):
        mask[start:start + rng.integers(10, 300)] = True
    x[mask] = 0
    return x, mask


class TestSemivariogram:
    def test_fft_equals_direct(self, gappy):
        x, mask = gappy
        a = semivariogram(x, 500, mask=mask, max_lag=4000)
        b = semivariogram(x, 500, mask=mask, max_lag=4000, method="direct")
        assert np.allclose(a.gamma, b.gamma, rtol=1e-8)
        assert np.array_equal(a.npairs, b.npairs)

    def test_brute_force_definition(self, gappy):
        x, mask = gappy
        vg = semivariogram(x, 500, mask=mask, max_lag=20)
        h = 7
        ok = ~mask[h:] & ~mask[:-h]
        assert vg.npairs[h - 1] == ok.sum()
        assert np.isclose(vg.gamma[h - 1], np.mean((x[h:] - x[:-h])[ok]**2) / 2)

    def test_lags_are_ms(self):
        vg = semivariogram(np.arange(100.), 500, max_lag="20 ms")
        assert np.allclose(vg.lags, np.arange(1, 11) * 2.)
        # linear trend: gamma(h) = h^2 / 2 (h in samples)
        assert np.allclose(vg.gamma, np.arange(1, 11)**2 / 2)

    def test_mask_sources_are_equivalent(self, gappy):
        x, mask = gappy
        ref = semivariogram(x, 500, mask=mask, max_lag=100)
        xnan = np.where(mask, np.nan, x)
        for sig in (xnan, np.ma.array(x, mask=mask)):
            vg = semivariogram(sig, 500, max_lag=100)
            assert np.allclose(vg.gamma, ref.gamma) and np.array_equal(vg.npairs, ref.npairs)

    def test_masked_values_do_not_enter(self, gappy):
        x, mask = gappy
        y = x.copy()
        y[mask] = 1e6
        assert np.allclose(semivariogram(x, 500, mask=mask).gamma,
                           semivariogram(y, 500, mask=mask).gamma)

    def test_lag_without_pairs_is_nan(self):
        x = np.ma.array(np.arange(100.), mask=np.arange(100) % 2 == 1)
        vg = semivariogram(x, 100, max_lag=40)
        assert np.isnan(vg.gamma[0]) and vg.npairs[0] == 0
        assert np.isclose(vg.gamma[1], 2.)

    def test_no_valid_samples(self):
        vg = semivariogram(np.full(50, np.nan), 100, max_lag=100)
        assert np.isnan(vg.gamma).all() and not vg.npairs.any()

    def test_max_lag_is_limited_to_signal(self):
        assert len(semivariogram(np.arange(20.), 100, max_lag="1 min").lags) == 19

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            semivariogram(np.arange(20.), 100, method="magic")


def test_plot(gappy):
    x, mask = gappy
    vg = semivariogram(x, 500, mask=mask)
    ax = plot_variogram({"a": vg, "b": vg}, units="sec")
    assert len(ax.get_legend().get_texts()) == 2
    ax = plot_variogram(vg, scale="linear", normalize=True)
    assert ax.get_legend() is None
    plt.close("all")
