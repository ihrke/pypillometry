"""Logging control (pypillometry.logging)."""
import contextlib

import numpy as np
from loguru import logger

import pypillometry as pp


@contextlib.contextmanager
def logged(level="WARNING"):
    """Collect the (function, message) of the records logged by loguru."""
    records = []
    handler = logger.add(lambda m: records.append((m.record["function"], m.record["message"])),
                         level=level)
    try:
        yield records
    finally:
        logger.remove(handler)


def warn(d):
    """Run something that logs a warning (no baseline estimated)."""
    d.pupil.estimate_response(npar=10, tmax=900, inplace=False)


def test_nologging_restores_logging():
    # nologging() read a loguru cache to decide whether to re-enable
    # logging, so logging could stay disabled after it
    rng = np.random.default_rng(1)
    d = pp.EyeData(left_pupil=rng.normal(size=1000), sampling_rate=100,
                   event_onsets=[1000, 5000], event_labels=["a", "a"])
    for _ in range(2):
        with pp.nologging():
            with pp.nologging():
                pass
            with logged() as records:
                warn(d)
            assert records == []
        with logged() as records:
            warn(d)
        assert records, "logging not restored after nologging()"


def test_debug_log_names_channels():
    d = pp.EyeData(left_pupil=np.arange(100.), right_pupil=np.arange(100.),
                   sampling_rate=100)
    with pp.loglevel("DEBUG"), logged("DEBUG") as records:
        d.scale(eyes="right")
        d.pupil.lowpass_filter(3, eyes="left")
    assert ("scale", "channels: right_pupil") in records
    assert ("lowpass_filter", "channels: left_pupil") in records
