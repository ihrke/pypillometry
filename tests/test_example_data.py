import os
import numpy as np
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

    def test_source_and_filetype(self):
        for key, info in example_data.example_datasets.items():
            self.assertIn(info["source"], {"OSF", "figshare", "local", "ssh"}, key)
            self.assertIn(info["filetype"], {"edf", "hdf5", "pkl", "asc"}, key)

    def test_descriptions_have_spaces(self):
        """Concatenated description strings are separated by spaces"""
        for key, info in example_data.example_datasets.items():
            self.assertNotRegex(info["description"], r"\.[A-Z]", key)
            self.assertNotIn("movementsbetween", info["description"], key)

    def test_get_rlmw_002(self):
        """Full dataset is named after the dataset (download mocked)"""
        short = {"samples": "002_rlmw_samples_short.asc",
                 "events": "002_rlmw_events_short.asc"}
        urls = {example_data.example_datasets["rlmw_002"]["files"][k]: v
                for k, v in short.items()}

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

    def test_get_nback_001(self):
        """Loader reads the pypil file and merges the behavioural log
        (figshare mocked)"""
        info = example_data.example_datasets["nback_001"]
        labels = ["experiment_start", "run-1_scr-grey_adapton",
                  "run-1_scr-grey_block-1_task-0b_trial-1_fix",
                  "run-1_scr-grey_block-1_task-0b_trial-1_stim_target-1",
                  "run-1_scr-grey_block-1_task-0b_trial-1_resp_acc-1",
                  "run-1_scr-grey_block-1_task-0b_trial-2_fix",
                  "run-1_scr-grey_block-1_task-0b_trial-2_stim_target-0",
                  "run-1_scr-grey_block-2_task-p_trial-1_fix"]
        tx = np.arange(0, 10000, 2.0)
        log = ("# N-back Task Log File\n# Ambient Light: High\n#\n"
               "subj,session,ambient,run,screen,block,task,trial,stimulus,target,"
               "response,accuracy,rt,fix_onset,stim_onset,resp_onset\n"
               "001,1,High,1,grey,1,0b,1,3,1,1,1,0.5,1.0,1.5,2.0\n"
               "001,1,High,1,grey,1,0b,2,7,0,0,1,,3.0,3.5,\n"
               "001,1,High,1,grey,2,p,1,4,0,0,,,5.0,5.5,\n").encode()
        with tempfile.TemporaryDirectory() as cache:
            src = os.path.join(cache, "src.pypil")
            pp.EyeData(time=tx, left_pupil=np.ones_like(tx),
                       event_onsets=np.arange(len(labels)) * 1000.0,
                       event_labels=labels, sampling_rate=500).write_file(src)
            with open(src, "rb") as f:
                content = {"data": f.read(), "behaviour": log}
            listing = {info["files"][k]: {"id": k, "download_url": f"https://x/files/{k}",
                                          "size": len(v), "md5": ""}
                       for k, v in content.items()}
            downloaded = []

            def fake_download(url, fname, **kwargs):
                downloaded.append(url)
                with open(fname, "wb") as f:
                    f.write(content[url.rsplit("/", 1)[1]])

            with mock.patch.object(pp.io, "get_figshare_files",
                                   return_value=listing) as mock_list, \
                 mock.patch.object(pp.io, "download", fake_download):
                d = pp.get_example_data("nback_001", cache_dir=cache)
                pp.get_example_data("nback_001", cache_dir=cache)
            self.assertEqual(len(downloaded), 2)  # second call: cached
            mock_list.assert_called_with(str(info["figshare"]["id"]),
                                         info["figshare"]["version"])
        self.assertEqual(d.name, "nback_001")
        # header lines without "key: value" are skipped
        self.assertEqual(d.info["behaviour_log"], {"Ambient Light": "High"})
        meta = d.events.meta
        np.testing.assert_array_equal(meta["task"],
                                      ["", "", "0b", "0b", "0b", "0b", "0b", "p"])
        np.testing.assert_array_equal(meta["stimulus"],
                                      [np.nan, np.nan, 3, 3, 3, 7, 7, 4])
        np.testing.assert_array_equal(meta["rt"],
                                      [np.nan, np.nan, .5, .5, .5, np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(meta["accuracy"][2:], [1, 1, 1, 1, 1, np.nan])

    def test_default_cache_dir(self):
        with mock.patch.dict("os.environ", {"XDG_CACHE_HOME": "/some/cache"}):
            self.assertEqual(example_data._user_cache_dir(),
                             os.path.join("/some/cache", "pypillometry"))


if __name__ == '__main__':
    unittest.main()
