"""Block means with counts, Welch spectrum with gaps, pooled variogram."""
import numpy as np
import pytest

from pypillometry.signal import (block_mean, coarsen, semivariogram, spectrum,
                                 plot_spectrum, Spectrum)
from pypillometry.signal import gaussmarkov as gm


def test_block_mean_counts_and_min_frac():
    x = np.arange(12, dtype=float)
    x[[1, 2, 3]] = np.nan           # block 0 has 1 of 4 valid samples
    mask = np.zeros(12, dtype=bool)
    mask[4] = True                  # block 1 has 3 of 4
    y, n = block_mean(x, 4, mask=mask)
    assert n.tolist() == [1, 3, 4]
    assert np.isnan(y[0]) and y[1] == 6.0 and y[2] == 9.5
    y, n = block_mean(x, 4, mask=mask, min_frac=0)
    assert y[0] == 0.0
    # incomplete last block is dropped; masked arrays work
    y, n = block_mean(np.ma.masked_invalid(x[:10]), 4)
    assert len(y) == 2 and n.tolist() == [1, 4]
    with pytest.raises(ValueError):
        block_mean(x, 0)


def test_coarsen_equals_block_mean_of_raw():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    x[rng.random(4000) < 0.3] = np.nan
    y, n = block_mean(x, 5, min_frac=0)
    yc, nc = coarsen(y, n, 4, block=5)
    yd, nd = block_mean(x, 20)
    assert np.array_equal(nc, nd)
    assert np.allclose(yc, yd, equal_nan=True)


def test_spectrum_matches_model_and_handles_gaps():
    model = gm.GaussMarkovModel([gm.Matern(3., 800.), gm.Matern(2., 20000.)], noise=0.3)
    y = gm.simulate(model, 100000, fs=20, rng=1)
    sp = spectrum(y, 20, window="250 s")
    assert isinstance(sp, Spectrum) and sp.window == 250000.0
    assert np.isclose(sp.freqs[0], 1 / 250) and np.isclose(sp.freqs[-1], 10)
    coarse = sp.at_log_freqs(5)
    theory = model.psd(coarse.freqs) + model.noise**2 / 10
    assert np.allclose(coarse.power[3:], theory[3:], rtol=0.25)
    # the integral of the spectrum is the variance
    assert np.isclose(np.sum(sp.power) * sp.freqs[0], np.var(y), rtol=0.15)
    # windows with many missing samples are skipped, others bridged
    y2 = y.copy()
    y2[20000:40000] = np.nan
    y2[np.random.default_rng(0).random(len(y)) < 0.02] = np.nan
    sp2 = spectrum(y2, 20, window="250 s")
    assert 0 < sp2.nwindows < sp.nwindows
    c2 = sp2.at_log_freqs(5)
    assert np.allclose(c2.power[3:-3], theory[3:-3], rtol=0.35)
    # windows stay within the segments
    seg = spectrum(y, 20, window="250 s", segments=[(0, 20000), (50000, 100000)])
    assert seg.nwindows == 7 + 19
    none = spectrum(np.full(1000, np.nan), 20, window="10 s")
    assert none.nwindows == 0 and np.isnan(none.power).all()
    with pytest.raises(ValueError):
        spectrum(y, 20, overlap=1.0)


def test_plot_spectrum():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.random.default_rng(0).normal(size=5000)
    sp = spectrum(x, 100, window="5 s")
    fig, ax = plt.subplots()
    assert plot_spectrum({"a": sp, "b": sp}, ax=ax) is ax
    assert len(ax.lines) == 2
    plt.close(fig)


def test_variogram_pooled_within_segments():
    rng = np.random.default_rng(2)
    model = gm.GaussMarkovModel([gm.Matern(1., 500.)], noise=0.1)
    y = gm.simulate(model, 6000, fs=20, rng=3)
    y[rng.random(6000) < 0.1] = np.nan
    one = semivariogram(y, 20, max_lag=20000)
    same = semivariogram(y, 20, max_lag=20000, segments=[(0, 6000)])
    assert np.allclose(one.gamma, same.gamma) and np.array_equal(one.npairs, same.npairs)
    # a jump of the level between two segments does not enter
    y2 = y.copy()
    y2[3000:] += 50
    segs = [(0, 3000), (3000, 6000)]
    pooled = semivariogram(y2, 20, max_lag=20000, segments=segs)
    ref = semivariogram(y, 20, max_lag=20000, segments=segs)
    assert np.allclose(pooled.gamma, ref.gamma)
    assert semivariogram(y2, 20, max_lag=20000).gamma[-1] > 10 * pooled.gamma[-1]
    assert np.isclose(pooled.variance, ref.variance)
    direct = semivariogram(y2, 20, max_lag=2000, segments=segs, method="direct")
    assert np.allclose(direct.gamma, pooled.gamma[:len(direct.gamma)])
    # the largest lag is limited by the longest segment
    assert len(semivariogram(y, 20, max_lag=1e9, segments=segs).lags) == 2999


def test_variogram_at_log_lags():
    x = np.random.default_rng(0).normal(size=5000)
    vg = semivariogram(x, fs=1000, max_lag=1000)
    lg = vg.at_log_lags(per_decade=4)
    assert len(lg.lags) == 13 and lg.lags[0] == 1 and lg.lags[-1] == 1000
    assert np.all(np.isin(lg.lags, vg.lags))
    assert len(vg.at_log_lags(per_decade=4, min_pairs=4500).lags) < 13
