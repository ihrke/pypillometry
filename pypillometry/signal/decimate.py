"""
decimate.py
===========

Reducing the sampling rate of a signal with missing samples by block means
that keep track of how many samples went into each mean.

Unlike low-pass filtering followed by decimation, block means need no
interpolation of the gaps: a block is either measured (the mean of its valid
samples) or missing, and the number of valid samples says how well it was
measured.

Experimental: the API may change.
"""
import numpy as np

from .variogram import _valid_data

__all__ = ["block_mean", "coarsen"]


def block_mean(signal, factor, mask=None, min_frac=0.5):
    """
    Means of the valid samples in consecutive blocks of `factor` samples.

    .. math::

        n_t = \\sum_{j \\in \\text{block } t} v_j, \\qquad
        \\bar x_t = \\frac{1}{n_t} \\sum_{j \\in \\text{block } t} v_j x_j

    with :math:`v_j = 1` for a valid sample and 0 otherwise.

    Parameters
    ----------
    signal : np.ndarray or np.ma.MaskedArray
        the signal; masked, NaN and inf samples are invalid
    factor : int
        number of samples per block; samples after the last complete block
        are dropped
    mask : np.ndarray, optional
        boolean array, True for additional invalid samples
    min_frac : float, optional
        a block is missing (NaN) unless at least this fraction of its
        samples is valid (default 0.5; 0: one valid sample is enough)

    Returns
    -------
    y : np.ndarray
        mean of the valid samples of each block (NaN where missing)
    n : np.ndarray
        number of valid samples of each block (int; also for the blocks
        that are missing because of `min_frac`)

    Examples
    --------
    >>> x = np.array([1., 2., 3., np.nan, 5., np.nan, np.nan, np.nan, 9.])
    >>> y, n = block_mean(x, 4)
    >>> y
    array([ 2., nan])
    >>> n
    array([3, 1])
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be a positive integer, got {factor}")
    x, v = _valid_data(signal, mask)
    m = len(x) // factor
    X = x[:m * factor].reshape(m, factor)
    V = v[:m * factor].reshape(m, factor)
    n = V.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = X.sum(axis=1) / n
    y[(n == 0) | (n < min_frac * factor)] = np.nan
    return y, n


def coarsen(y, n, factor, block=1, min_frac=0.5):
    """
    Coarser block means from block means (see :func:`block_mean`): means of
    `factor` consecutive blocks, weighted by their numbers of valid samples.

    Parameters
    ----------
    y, n : np.ndarray
        block means (NaN where missing) and numbers of valid samples
    factor : int
        number of blocks per new block
    block : int, optional
        number of raw samples per block of the input (for `min_frac`);
        default 1
    min_frac : float, optional
        a new block is missing unless at least this fraction of its
        ``factor * block`` raw samples is valid (default 0.5)

    Returns
    -------
    y, n : np.ndarray
        as for :func:`block_mean`, for blocks of ``factor * block`` raw samples

    Examples
    --------
    >>> y, n = coarsen(np.array([1., 3., np.nan, 4.]), np.array([4, 2, 1, 4]), 2, block=4)
    >>> y, n
    (array([1.66666667, 4.        ]), array([6, 4]))
    """
    factor = int(factor)
    y, n = np.asarray(y, dtype=float), np.asarray(n)
    m = len(y) // factor
    Y = y[:m * factor].reshape(m, factor)
    N = np.where(np.isfinite(Y), n[:m * factor].reshape(m, factor), 0)
    nn = N.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.isfinite(Y), Y * N, 0.0).sum(axis=1) / nn
    out[(nn == 0) | (nn < min_frac * factor * block)] = np.nan
    return out, nn
