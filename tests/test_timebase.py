"""Time vector, sampling rate and the operations that change the samples."""
import unittest

import numpy as np

import pypillometry as pp


def make(n=1000, fs=1000.):
    """PupilData with `n` samples at `fs` Hz (sample i at i*1000/fs ms)."""
    tx = np.arange(n) * 1000. / fs
    return pp.PupilData(left_pupil=np.full(n, 10.), time=tx, sampling_rate=fs)


def actual_fs(d):
    return 1000. / np.median(np.diff(d.tx))


class TestDownsampleRate(unittest.TestCase):

    def test_integer_factor(self):
        x = make().downsample(250, inplace=False)
        self.assertEqual(x.fs, 250.)
        self.assertEqual(len(x), 250)

    def test_non_integer_ratio_stores_actual_rate(self):
        # 1000/300 -> factor 3 -> 333.3 Hz, not 300
        x = make().downsample(300, inplace=False)
        self.assertAlmostEqual(x.fs, 1000. / 3)
        self.assertAlmostEqual(x.fs, actual_fs(x))
        # 1000/400 -> factor 2 -> 500 Hz
        x = make().downsample(400, inplace=False)
        self.assertEqual(x.fs, 500.)
        self.assertAlmostEqual(x.fs, actual_fs(x))

    def test_factor(self):
        x = make().downsample(4, dsfac=True, inplace=False)
        self.assertEqual(x.fs, 250.)
        self.assertAlmostEqual(x.fs, actual_fs(x))


if __name__ == "__main__":
    unittest.main()
