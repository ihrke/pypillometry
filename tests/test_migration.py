"""
Migration of saved states from format 2 to 3 (`state._migrate_2_to_3`).

The fixture files in ``tests/data/format2`` were written by pypillometry
before the 2.2 refactor; ``tests/test_format2_fixtures.py`` checks the loaded
objects against their manifest. The tests here check the migrated states.
"""
import contextlib
import os

import numpy as np
import pytest
from loguru import logger

import pypillometry as pp
from pypillometry import hdf5, state

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "format2")


def format2_state(name):
    """The state in a format-2 HDF5 fixture file (not migrated)."""
    st = hdf5.read_state(os.path.join(FIXTURE_DIR, name))
    assert st["format_version"] == 2
    return st


@contextlib.contextmanager
def warnings_logged():
    """Collect the warnings logged by loguru."""
    messages = []
    handler = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler)


class TestSamplingRate:

    def test_fs_fits_tx_unchanged(self):
        with warnings_logged() as logs:
            st = state.migrate(format2_state("pupildata.h5"), "PupilData")
        assert st["fs"] == 500.
        assert not [m for m in logs if "sampling rate" in m]

    def test_fs_corrected_from_tx(self):
        # as saved after downsample(fsd) with a non-integer fs/fsd
        st = format2_state("pupildata.h5")
        st["fs"] = 300.
        with warnings_logged() as logs:
            d = state.from_state(st)
        assert d.fs == pytest.approx(500.)
        assert any("sampling rate (300 Hz)" in m for m in logs)
