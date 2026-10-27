"""Tests for loading a study from a local directory or an SSH URL."""
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pypillometry import study as ppstudy
from tests.test_io_ssh import SSHTestCase


CONFIG = """
raw_data = {
    "s1": {"data": "s1/data.txt", "group": "control"},
    "s2": {"data": "s2/data.txt", "group": "patient"},
}

def read_subject(info):
    with open(info["data"]) as f:
        return f.read(), info["group"]
"""


class TestLoadStudySSH(SSHTestCase):
    def setUp(self):
        super().setUp()
        study = os.path.join(self.server_root, "study")
        for sid in ["s1", "s2"]:
            os.makedirs(os.path.join(study, sid))
            with open(os.path.join(study, sid, "data.txt"), "w") as f:
                f.write(f"data {sid}")
        with open(os.path.join(study, "pypillometry_conf.py"), "w") as f:
            f.write(CONFIG)
        self.study = study
        # run in an empty directory so no config file is found in the cwd
        self.cwd = os.getcwd()
        self.workdir = tempfile.mkdtemp()
        os.chdir(self.workdir)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.workdir, ignore_errors=True)
        super().tearDown()

    def test_local(self):
        data, config = ppstudy.load_study_local(self.study)
        self.assertEqual(data["s1"], ("data s1", "control"))
        self.assertEqual(self.server.calls, [])

    def test_ssh(self):
        data, config = ppstudy.load_study_local("ssh://server/study/", cache_dir=self.cache)
        self.assertEqual(data["s1"], ("data s1", "control"))
        self.assertEqual(data["s2"], ("data s2", "patient"))
        self.assertTrue(os.path.exists(os.path.join(self.cache, "server", "study", "s2", "data.txt")))

    def test_ssh_subset(self):
        data, _ = ppstudy.load_study_local("ssh://server/study", subjects=["s2"], cache_dir=self.cache)
        self.assertEqual(list(data), ["s2"])
        self.assertFalse(os.path.exists(os.path.join(self.cache, "server", "study", "s1")))

    def test_ssh_autodelete(self):
        data, _ = ppstudy.load_study_local("ssh://server/study/", cache_dir=self.cache, autodelete=True)
        self.assertEqual(data["s1"], ("data s1", "control"))
        self.assertFalse(os.path.exists(os.path.join(self.cache, "server", "study", "s1", "data.txt")))

    def test_ssh_temp_dir_removed(self):
        with patch("tempfile.mkdtemp", return_value=os.path.join(self.cache, "tmp")):
            os.makedirs(os.path.join(self.cache, "tmp"))
            data, _ = ppstudy.load_study_local("ssh://server/study/", autodelete=True)
        self.assertEqual(len(data), 2)
        self.assertFalse(os.path.exists(os.path.join(self.cache, "tmp")))



if __name__ == "__main__":
    unittest.main()
