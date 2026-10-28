import os
import shutil
import tempfile
import unittest
from importlib.resources import files
from unittest import mock

import pypillometry as pp
from pypillometry import example_data

TEST_EDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test.edf")


class TestExampleData(unittest.TestCase):
    def test_every_dataset_has_loader(self):
        for key in example_data.example_datasets:
            self.assertTrue(callable(getattr(example_data, f"get_{key}", None)), key)

    def test_get_rlmw_010_edf(self):
        """Loader reads the downloaded EDF file (download mocked)"""
        edf = TEST_EDF

        def fake_download(url, fname, **kwargs):
            shutil.copy(edf, fname)

        with tempfile.TemporaryDirectory() as cache, \
             mock.patch.object(pp.io, "download", fake_download):
            d = pp.get_example_data("rlmw_010_edf", cache_dir=cache)
        self.assertIsInstance(d, pp.EyeData)
        self.assertEqual(d.name, "rlmw_010")
        self.assertAlmostEqual(d.experimental_setup.eye_to_screen_center, 600.0)

    def test_descriptions_have_spaces(self):
        """Concatenated description strings are separated by spaces"""
        for key, info in example_data.example_datasets.items():
            self.assertNotRegex(info["description"], r"\.[A-Z]", key)
            self.assertNotIn("movementsbetween", info["description"], key)

    def test_get_rlmw_002(self):
        """Full dataset is named after the dataset (download mocked)"""
        short = {"samples_asc": "002_rlmw_samples_short.asc",
                 "events_asc": "002_rlmw_events_short.asc"}
        urls = {example_data.example_datasets["rlmw_002"][k]: v for k, v in short.items()}

        downloaded = []

        def fake_download(url, fname, **kwargs):
            downloaded.append(url)
            shutil.copy(files('pypillometry.data').joinpath(urls[url]), fname)

        with tempfile.TemporaryDirectory() as cache, \
             mock.patch.object(pp.io, "download", fake_download):
            d = pp.get_example_data("rlmw_002", cache_dir=cache)
            self.assertEqual(len(downloaded), 2)
            # second call uses the cached files
            pp.get_example_data("rlmw_002", cache_dir=cache)
            self.assertEqual(len(downloaded), 2)
        self.assertEqual(d.name, "rlmw_002")

    def test_default_cache_dir(self):
        with mock.patch.dict("os.environ", {"XDG_CACHE_HOME": "/some/cache"}):
            self.assertEqual(example_data._user_cache_dir(),
                             os.path.join("/some/cache", "pypillometry"))


if __name__ == '__main__':
    unittest.main()
