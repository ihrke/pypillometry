"""
group.py
========

:class:`TimescaleGroup`: group-level (hierarchical) fit of the
:class:`~pypillometry.TimescaleModel` of several recordings.
"""
import numpy as np
from loguru import logger

from ..signal import gaussmarkov as gm
from ..units import parse_time
from .model import TimescaleModel
from .plots import TimescaleGroupPlots

__all__ = ["TimescaleGroup"]


class TimescaleGroup:
    """
    Group of recordings whose timescale models are estimated together
    (experimental).

    The parameters of the recordings are treated as draws from a normal
    distribution with a group mean and a (diagonal) covariance, which are
    estimated together with the parameters of each recording (empirical
    Bayes, see :mod:`pypillometry.signal.gaussmarkov.hierarchical`). The
    group mean is the typical mix of time scales; the estimate of each
    recording is shrunk towards it where the recording itself is
    uninformative (the variance of single atoms), and stays where the
    data determine it (the fast component, the noise, the total slow
    variance).

    Parameters
    ----------
    models : dict or list
        the recordings: :class:`TimescaleModel` objects, or
        :class:`~pypillometry.EyeData` objects, from which models are made
        with `**kwargs`. The keys of a dict are the names of the
        recordings (for a list: the names of the objects). All models
        need the same structure (time scales, binocular or not).
    **kwargs
        arguments of :class:`TimescaleModel` for the EyeData objects
        (e.g., ``eyes="left"``, ``binocular=True``)

    Attributes
    ----------
    models : dict
        name -> :class:`TimescaleModel`. After :meth:`fit`, their
        estimates are the shrunk ones (``model.fits``) and the estimates
        without pooling are in ``model.individual``.
    bank : TimescaleBank
        the common structure

    Notes
    -----
    Every unit of every model (each eye of each recording, unless the
    models are binocular or use one eye) counts as one independent member
    of the group.

    Examples
    --------
    >>> g = pp.TimescaleGroup({"s1": d1, "s2": d2, "s3": d3}, eyes="left")  # doctest: +SKIP
    >>> g.fit()  # doctest: +SKIP
    >>> g.group_summary()  # doctest: +SKIP
    >>> g.models["s1"].plot.decomposition()  # doctest: +SKIP
    >>> new = pp.TimescaleModel(d4, eyes="left").fit(prior=g.prior)  # doctest: +SKIP
    """

    def __init__(self, models, **kwargs):
        if not isinstance(models, dict):
            models = list(models)
            names = [m.name for m in models]
            if len(set(names)) != len(names):
                raise ValueError("the names of the recordings are not unique; pass a dict")
            models = dict(zip(names, models))
        self.models = {}
        for name, m in models.items():
            if not isinstance(m, TimescaleModel):
                m = TimescaleModel(m, name=name, **kwargs)
            self.models[name] = m
        if len(self.models) < 2:
            raise ValueError("a group needs at least two recordings")
        first = next(iter(self.models.values()))
        self.bank = first.bank
        for name, m in self.models.items():
            # channels of binocular models are named after the eyes
            same = (m.bank.timescales, m.bank.nu, m.bank.v0, m.bank.n_channels) == \
                (self.bank.timescales, self.bank.nu, self.bank.v0, self.bank.n_channels)
            if not same or m.bank.names != self.bank.names:
                raise ValueError(f"the model of {name!r} has another structure "
                                 f"({m.bank}) than the first one ({self.bank})")
        self.group = None

    @classmethod
    def from_files(cls, files, preprocess=None, **kwargs):
        """
        Group from files that are read one at a time, so that only the
        reduced data of each recording stay in memory.

        Parameters
        ----------
        files : dict or list of str
            the files (read with :func:`~pypillometry.read_file`); the keys
            of a dict are the names of the recordings (for a list: the file
            names without directory and extension)
        preprocess : callable, optional
            function that takes the :class:`~pypillometry.EyeData` object of
            a file and returns the object to model, e.g. with blinks
            detected and the periods outside the task excluded
        **kwargs
            arguments of :class:`TimescaleModel`

        Returns
        -------
        TimescaleGroup
            its models keep no reference to the data; pass the data to
            :meth:`TimescaleModel.interpolate` and
            :meth:`TimescaleModel.store`
        """
        import os
        from ..io import read_file
        if not isinstance(files, dict):
            files = {os.path.splitext(os.path.basename(str(f)))[0]: f for f in files}
        models = {}
        for name, fname in files.items():
            d = read_file(fname)
            if preprocess is not None:
                d = preprocess(d)
            models[name] = TimescaleModel(d, name=name, keep_data=False, **kwargs)
        return cls(models)

    def __repr__(self):
        state = "not fitted" if self.group is None else "fitted"
        return (f"TimescaleGroup({len(self.models)} recordings, {len(self._members())} "
                f"members, {self.bank}; {state})")

    def _members(self):
        """(name of the recording, unit) of each member of the group"""
        return [(name, u) for name, m in self.models.items() for u in m.units]

    def _group(self):
        if self.group is None:
            raise ValueError("the group is not fitted yet; call fit() first")
        return self.group

    def fit(self, n_iter=5, tol=0.01, cov="diag", n_jobs=None):
        """
        Estimate the group distribution and the parameters of all recordings.

        Parameters
        ----------
        n_iter : int
            maximum number of iterations (default 5)
        tol : float
            stop when the group mean changes by less than this
        cov : {"diag", "full"}
            covariance of the parameters between recordings: independent
            (default) or with correlations (for large groups)
        n_jobs : int, optional
            number of parallel processes (default: one per core but one)

        Returns
        -------
        TimescaleGroup
            the group itself (fitted)

        Notes
        -----
        Each iteration fits every recording once (about a minute per 20
        minutes of data and process).
        """
        members = self._members()
        data = [(self.models[n]._Y(u), self.models[n].fs, self.models[n].segments)
                for n, u in members]
        logger.info(f"fitting a group of {len(members)} members")
        self.group = gm.fit_group(self.bank, data, n_iter=n_iter, tol=tol, cov=cov,
                                  n_jobs=n_jobs)
        for (name, unit), shrunk, ind in zip(members, self.group.fits,
                                             self.group.individual):
            model = self.models[name]
            # the fits refer to the bank of the group; the models share its structure
            shrunk.bank = ind.bank = model.bank
            model._set_fit(unit, shrunk)
            model.individual[unit] = ind
        return self

    @property
    def prior(self):
        """
        The group distribution as a prior for the fit of other recordings
        of the same kind, ``TimescaleModel(d).fit(prior=g.prior)``.
        """
        return self._group().prior

    def group_summary(self):
        """
        Table of the group distribution.

        Returns
        -------
        pandas.DataFrame
            one row per parameter: ``mean`` and ``sd`` of the entries of
            the parameter vector (logs, and :math:`u = \\log(v + v_0)` for
            the atoms) and ``typical``, ``low`` and ``high``: the group
            mean and the mean minus/plus one standard deviation on the
            natural scale (standard deviations in log-%, length scales in
            ms)
        """
        import pandas as pd
        g = self._group()
        nat = lambda th: self.bank.params(th)
        rows = {}
        for i, (name, key) in enumerate(zip(self.bank.names, nat(g.mu))):
            e = np.zeros(len(g.mu))
            e[i] = g.sd[i]
            rows[key] = dict(parameter=name, mean=g.mu[i], sd=g.sd[i], typical=nat(g.mu)[key],
                             low=nat(g.mu - e)[key], high=nat(g.mu + e)[key])
        return pd.DataFrame.from_dict(rows, orient="index")

    def summary(self, cutoff="10 s", individual=False):
        """
        Table of the estimates of all recordings.

        Parameters
        ----------
        cutoff : float, str or Quantity
            time scale from which on the atoms count as slow (plain numbers
            are ms; default 10 s)
        individual : bool
            the estimates without pooling instead of the shrunk ones

        Returns
        -------
        pandas.DataFrame
            one row per recording and unit (see
            :meth:`TimescaleModel.summary`)
        """
        import pandas as pd
        self._group()
        c = float(parse_time(cutoff, to="ms"))
        rows = []
        for name, unit in self._members():
            m = self.models[name]
            fit = (m.individual if individual else m.fits)[unit]
            rows.append({"recording": name, "unit": unit, **fit.params,
                         "slow_sd": self.bank.slow_sd(fit.theta, c), "n_obs": fit.n_obs,
                         "loglik": fit.loglik})
        return pd.DataFrame(rows).set_index(["recording", "unit"])

    @property
    def plot(self):
        """plots of the group (:class:`~pypillometry.timescale.plots.TimescaleGroupPlots`)"""
        return TimescaleGroupPlots(self)
