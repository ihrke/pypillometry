import shutil
import unittest
from importlib.resources import files
from unittest import mock

import pypillometry as pp
from pypillometry import example_data


class TestExampleData(unittest.TestCase):
    def test_every_dataset_has_loader(self):
        for key in example_data.example_datasets:
            self.assertTrue(callable(getattr(example_data, f"get_{key}", None)), key)

    def test_get_rlmw_010_edf(self):
        """Loader reads the downloaded EDF file (download mocked)"""
        edf = files('pypillometry.data').joinpath('test.edf')

        def fake_download(url, fname, **kwargs):
            shutil.copy(edf, fname)

        with mock.patch.object(example_data, "download", fake_download):
            d = pp.get_example_data("rlmw_010_edf")
        self.assertIsInstance(d, pp.EyeData)
        self.assertEqual(d.name, "rlmw_010")
        self.assertAlmostEqual(d.experimental_setup.eye_to_screen_center, 600.0)


if __name__ == '__main__':
    unittest.main()
