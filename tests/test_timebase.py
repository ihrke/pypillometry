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


class TestConstructorTime(unittest.TestCase):

    def test_default_time_vector(self):
        # without `time`, sample i is at i*1000/fs ms
        d = pp.PupilData(left_pupil=np.ones(5), sampling_rate=1000)
        np.testing.assert_array_equal(d.tx, [0., 1., 2., 3., 4.])
        d = pp.PupilData(left_pupil=np.ones(4), sampling_rate=300)
        np.testing.assert_allclose(d.tx, np.arange(4) * 1000. / 300)


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


class TestDownsampleArguments(unittest.TestCase):

    def test_bad_rates(self):
        d = make()
        for fsd in (0, -10, 2000):
            with self.assertRaisesRegex(ValueError, "sampling rate in Hz"):
                d.downsample(fsd, inplace=False)
        with self.assertRaisesRegex(TypeError, "fsd must be a finite number"):
            d.downsample("100 Hz", inplace=False)
        for fsd in (np.inf, np.nan):
            with self.assertRaisesRegex(TypeError, "fsd must be a finite number"):
                d.downsample(fsd, inplace=False)

    def test_bad_factors(self):
        d = make()
        for fac in (0, -2, 2.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                d.downsample(fac, dsfac=True, inplace=False)

    def test_same_rate_is_a_copy(self):
        d = make()
        x = d.downsample(1000, inplace=False)
        self.assertEqual(x.fs, 1000.)
        np.testing.assert_array_equal(x.tx, d.tx)

    def test_float_noise_in_ratio(self):
        # 1000/(1000/15) is 14.999999999999998 in floating point
        d = make()
        x = d.downsample(1000. / 15, inplace=False)
        self.assertEqual(len(x), int(np.ceil(len(d) / 15)))
        self.assertAlmostEqual(x.fs, 1000. / 15)


if __name__ == "__main__":
    unittest.main()
