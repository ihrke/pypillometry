"""
table.py
========

:class:`Table`: results that are tables (e.g., :meth:`EyeData.layer_stats
<pypillometry.EyeData.layer_stats>`), as a dict of columns that displays as a
:class:`pandas.DataFrame`.
"""
import numpy as np

__all__ = ["Table"]


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
