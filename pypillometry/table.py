"""
table.py
========

Results that are displayed as tables (:class:`pandas.DataFrame`) but are
used as dicts:

- :class:`Table`: a dict of columns, for results with one row per event or
  trial (e.g., :meth:`Events.to_table <pypillometry.Events.to_table>`)
- :class:`RecordDict`: a dict of records (e.g., channel -> layer ->
  statistics in :meth:`EyeData.layer_stats
  <pypillometry.EyeData.layer_stats>`)
"""
import numpy as np

__all__ = ["Table", "RecordDict"]


class Table(dict):
    """
    A table: a dict of columns (column name -> numpy array), all of the same
    length (one value per row).

    It is a plain dict (``t["n"]``, ``t.keys()``, ``pd.DataFrame(t)``), with
    numpy arrays as columns, but it is shown as a table: as a
    :class:`pandas.DataFrame` in notebooks and with ``print()``.
    :meth:`to_pandas` converts it.

    Parameters
    ----------
    columns : dict, optional
        column name -> values (converted to numpy arrays)
    **kwargs
        more columns

    Raises
    ------
    ValueError
        if the columns differ in length

    Examples
    --------
    >>> import pypillometry as pp
    >>> t = pp.Table({"channel": ["left_pupil", "right_pupil"], "n": [14, 13]})
    >>> t["n"]
    array([14, 13])
    >>> t.nrows
    2
    >>> print(t)
           channel   n
    0   left_pupil  14
    1  right_pupil  13
    """

    def __init__(self, columns=None, **kwargs):
        super().__init__()
        for name, values in {**dict(columns or {}), **kwargs}.items():
            self[name] = np.asarray(values)
        lengths = {name: len(col) for name, col in self.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"the columns of a table must have the same length, got "
                             f"{', '.join(f'{k}: {n}' for k, n in lengths.items())}")

    @property
    def nrows(self) -> int:
        """Number of rows."""
        return len(next(iter(self.values()))) if self else 0

    def to_pandas(self):
        """
        The table as a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd
        return pd.DataFrame(dict(self))

    def __repr__(self) -> str:
        return repr(self.to_pandas())

    def _repr_html_(self) -> str:
        return self.to_pandas()._repr_html_()


class RecordDict(dict):
    """
    A dict of records that is displayed as a table.

    The keys name the rows: ``key -> record`` (one level, e.g. dataset name
    -> its properties) or ``key -> key -> record`` (two levels, e.g.
    channel -> layer -> statistics). A record is a dict of values (the
    columns of its row). It is used like any nested dict
    (``stats["left_pupil"]["blink"]["n"]``); only its display differs: in
    notebooks and with ``print()``, it is shown as a
    :class:`pandas.DataFrame` with one row per record (:meth:`to_pandas`).

    Parameters
    ----------
    data : dict, optional
        the records
    index_names : tuple of str, optional
        names of the key levels in the table (e.g., ``("channel",
        "layer")``)

    Examples
    --------
    >>> import pypillometry as pp
    >>> r = pp.RecordDict({"left_pupil": {"blink": {"n": 14}, "masked": {"n": 0}}},
    ...                   index_names=("channel", "layer"))
    >>> r["left_pupil"]["blink"]["n"]
    14
    >>> print(r)  # doctest: +NORMALIZE_WHITESPACE
                       n
    channel    layer    
    left_pupil blink   14
               masked   0
    """

    def __init__(self, data=None, index_names=None):
        super().__init__(data or {})
        self.index_names = None if index_names is None else tuple(index_names)

    def _rows(self) -> list:
        """(index tuple, record) of each row."""
        rows = []
        for key, value in self.items():
            if value and all(isinstance(v, dict) for v in value.values()):
                rows += [((key, key2), dict(rec)) for key2, rec in value.items()]
            else:
                rows.append(((key,), dict(value)))
        return rows

    def to_pandas(self):
        """
        The records as a :class:`pandas.DataFrame`, one row per record.

        Returns
        -------
        pandas.DataFrame
            indexed by the keys (a MultiIndex for two levels of keys)
        """
        import pandas as pd
        rows = self._rows()
        if not rows:
            return pd.DataFrame()
        keys = [k for k, _ in rows]
        if max(len(k) for k in keys) > 1:
            index = pd.MultiIndex.from_tuples(keys, names=self.index_names)
        else:
            index = pd.Index([k[0] for k in keys],
                             name=self.index_names[0] if self.index_names else None)
        return pd.DataFrame([r for _, r in rows], index=index)

    def __repr__(self) -> str:
        return repr(self.to_pandas())

    def _repr_html_(self) -> str:
        return self.to_pandas()._repr_html_()
