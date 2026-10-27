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


CONFIG_OSF = """
raw_data = {"s1": {"data": "s1/data.txt"}, "s2": {"data": "s2/data.txt"}}

def read_subject(info):
    with open(info["data"]) as f:
        return f.read()
"""


class TestLoadStudyOSF(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        content = {"pypillometry_conf.py": CONFIG_OSF.encode(),
                   "s1/data.txt": b"data s1", "s2/data.txt": b"data s2"}
        self.files = {"/" + k: {"size": 2 * 1024 * 1024, "download_url": k}
                      for k in content}

        class Response:
            status_code = 200
            headers = {}
            def __init__(self, data):
                self.content = data
            def iter_content(self, chunk_size):
                yield self.content

        self.get = lambda url, **kw: Response(content[url])

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def load(self, **kwargs):
        import io
        from contextlib import redirect_stdout
        from loguru import logger
        messages = []
        hid = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
        stdout = io.StringIO()
        try:
            with patch.object(ppstudy, "get_osf_project_files", return_value=self.files), \
                 patch.object(ppstudy.requests, "get", side_effect=self.get), \
                 redirect_stdout(stdout):
                data, _ = ppstudy.load_study_osf("abcde", self.path, **kwargs)
        finally:
            logger.remove(hid)
        return data, messages, stdout.getvalue()

    def test_logs_instead_of_print(self):
        data, messages, stdout = self.load()
        self.assertEqual(data["s2"], "data s2")
        self.assertEqual(stdout, "")
        self.assertTrue(any("total size: 4.0 MB" in m for m in messages))
        # nothing left to download: no size estimate
        _, messages, _ = self.load()
        self.assertFalse(any("total size" in m for m in messages))


if __name__ == "__main__":
    unittest.main()
