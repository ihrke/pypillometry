"""Tests for reading from ssh:// URLs (scp is replaced by a local copy)."""
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from importlib.resources import files

from pypillometry import io

DATA_DIR = str(files("pypillometry.data"))


class FakeServer:
    """Replaces `subprocess.run` for scp: `host:/path` is read from `root/path`."""

    def __init__(self, root):
        self.root = root
        self.calls = []

    def run(self, cmd, **kwargs):
        self.calls.append(cmd)
        src, dst = cmd[-2], cmd[-1]
        _, rpath = src.split(":", 1)
        local_src = os.path.join(self.root, rpath.lstrip("/"))
        if not os.path.exists(local_src):
            return subprocess.CompletedProcess(cmd, 1, "", f"scp: {rpath}: No such file or directory")
        if os.path.isdir(local_src):
            shutil.copytree(local_src, dst)
        else:
            shutil.copy(local_src, dst)
        return subprocess.CompletedProcess(cmd, 0, "", "")


class SSHTestCase(unittest.TestCase):
    def setUp(self):
        self.server_root = tempfile.mkdtemp()
        self.cache = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.server_root, "data", "sub"))
        with open(os.path.join(self.server_root, "data", "f.txt"), "w") as f:
            f.write("hello")
        with open(os.path.join(self.server_root, "data", "sub", "g.txt"), "w") as f:
            f.write("world")
        self.server = FakeServer(self.server_root)
        p1 = patch("subprocess.run", side_effect=self.server.run)
        p2 = patch("shutil.which", return_value="/usr/bin/scp")
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def tearDown(self):
        shutil.rmtree(self.server_root, ignore_errors=True)
        shutil.rmtree(self.cache, ignore_errors=True)


class TestParseSSHUrl(unittest.TestCase):
    def test_is_ssh_url(self):
        self.assertTrue(io.is_ssh_url("ssh://server/folder/f.edf"))
        self.assertFalse(io.is_ssh_url("https://server/f.edf"))
        self.assertFalse(io.is_ssh_url("/local/f.edf"))
        self.assertFalse(io.is_ssh_url(None))

    def test_parse(self):
        self.assertEqual(io._parse_ssh_url("ssh://server/folder/f.edf"),
                         ("server", None, "/folder/f.edf"))
        self.assertEqual(io._parse_ssh_url("ssh://me@server:2222/~/f.edf"),
                         ("me@server", 2222, "f.edf"))
        self.assertEqual(io._parse_ssh_url("ssh://server/folder/"),
                         ("server", None, "/folder/"))

    def test_parse_invalid(self):
        for url in ["ssh://server", "ssh://server/", "ssh:///f.edf", "http://server/f"]:
            with self.assertRaises(ValueError):
                io._parse_ssh_url(url)

    def test_cache_path(self):
        self.assertEqual(io._ssh_cache_path("ssh://srv/a/b.edf", "/c"),
                         os.path.join("/c", "srv", "a", "b.edf"))
        self.assertEqual(io._ssh_cache_path("ssh://srv:22/~/a/b.edf", "/c"),
                         os.path.join("/c", "srv_22", "~", "a", "b.edf"))


class TestSSHDownload(SSHTestCase):
    def test_download_file(self):
        fname = io.ssh_download("ssh://server/data/f.txt", cache_dir=self.cache)
        self.assertEqual(fname, os.path.join(self.cache, "server", "data", "f.txt"))
        with open(fname) as f:
            self.assertEqual(f.read(), "hello")
        cmd = self.server.calls[0]
        self.assertEqual(cmd[0], "scp")
        self.assertIn("BatchMode=yes", cmd)
        self.assertEqual(cmd[-2], "server:/data/f.txt")

    def test_port_and_user(self):
        io.ssh_download("ssh://me@server:2222/data/f.txt", cache_dir=self.cache)
        cmd = self.server.calls[0]
        self.assertEqual(cmd[cmd.index("-P") + 1], "2222")
        self.assertEqual(cmd[-2], "me@server:/data/f.txt")

    def test_temp_dir_when_no_cache_dir(self):
        fname = io.ssh_download("ssh://server/data/f.txt")
        try:
            self.assertTrue(os.path.exists(fname))
            self.assertIn("pypillometry_ssh_", fname)
        finally:
            shutil.rmtree(fname.split(os.sep + "server" + os.sep)[0], ignore_errors=True)

    def test_cached_file_is_reused(self):
        io.ssh_download("ssh://server/data/f.txt", cache_dir=self.cache)
        io.ssh_download("ssh://server/data/f.txt", cache_dir=self.cache)
        self.assertEqual(len(self.server.calls), 1)
        io.ssh_download("ssh://server/data/f.txt", cache_dir=self.cache, force_download=True)
        self.assertEqual(len(self.server.calls), 2)

    def test_download_folder(self):
        local = io.ssh_download("ssh://server/data/", cache_dir=self.cache)
        self.assertIn("-r", self.server.calls[0])
        self.assertTrue(os.path.isfile(os.path.join(local, "sub", "g.txt")))

    def test_force_download_folder(self):
        io.ssh_download("ssh://server/data/", cache_dir=self.cache)
        local = io.ssh_download("ssh://server/data/", cache_dir=self.cache, force_download=True)
        self.assertTrue(os.path.isfile(os.path.join(local, "sub", "g.txt")))
        self.assertFalse(os.path.exists(os.path.join(local, "data")))

    def test_failed_download_leaves_no_file(self):
        with self.assertRaises(RuntimeError):
            io.ssh_download("ssh://server/data/missing.txt", cache_dir=self.cache)
        self.assertFalse(os.path.exists(os.path.join(self.cache, "server", "data", "missing.txt")))

    def test_error(self):
        with self.assertRaisesRegex(RuntimeError, "No such file"):
            io.ssh_download("ssh://server/data/missing.txt", cache_dir=self.cache)


class TestLocalCopy(SSHTestCase):
    def test_local_path_passes_through(self):
        local = os.path.join(self.server_root, "data", "f.txt")
        with io.local_copy(local, autodelete=True) as fname:
            self.assertEqual(fname, local)
        self.assertTrue(os.path.exists(local))
        self.assertEqual(self.server.calls, [])

    def test_keep_by_default(self):
        with io.local_copy("ssh://server/data/f.txt", cache_dir=self.cache) as fname:
            pass
        self.assertTrue(os.path.exists(fname))

    def test_autodelete_in_cache_dir(self):
        with io.local_copy("ssh://server/data/f.txt", cache_dir=self.cache, autodelete=True) as fname:
            self.assertTrue(os.path.exists(fname))
        self.assertFalse(os.path.exists(fname))
        self.assertTrue(os.path.isdir(self.cache))

    def test_autodelete_removes_temp_dir(self):
        with io.local_copy("ssh://server/data/f.txt", autodelete=True) as fname:
            tmpdir = fname.split(os.sep + "server" + os.sep)[0]
            self.assertTrue(os.path.isdir(tmpdir))
        self.assertFalse(os.path.exists(tmpdir))


class TestReadEyelinkSSH(SSHTestCase):
    def setUp(self):
        super().setUp()
        shutil.copy(os.path.join(DATA_DIR, "test.edf"),
                    os.path.join(self.server_root, "data", "test.edf"))

    def test_read_eyelink(self):
        edf = io.read_eyelink("ssh://server/data/test.edf", cache_dir=self.cache)
        self.assertIn("samples", edf)
        self.assertTrue(os.path.exists(os.path.join(self.cache, "server", "data", "test.edf")))

    def test_read_eyelink_autodelete(self):
        io.read_eyelink("ssh://server/data/test.edf", cache_dir=self.cache, autodelete=True)
        self.assertFalse(os.path.exists(os.path.join(self.cache, "server", "data", "test.edf")))

    def test_from_eyelink(self):
        import pypillometry as pp
        d = pp.EyeData.from_eyelink("ssh://server/data/test.edf", autodelete=True)
        self.assertGreater(len(d.tx), 0)


class TestReadPickleSSH(SSHTestCase):
    def test_read_pickle(self):
        io.write_pickle({"a": 1}, os.path.join(self.server_root, "data", "obj.pkl"))
        obj = io.read_pickle("ssh://server/data/obj.pkl", cache_dir=self.cache, autodelete=True)
        self.assertEqual(obj, {"a": 1})
        self.assertFalse(os.path.exists(os.path.join(self.cache, "server", "data", "obj.pkl")))

    def test_from_file(self):
        import pypillometry as pp
        d = pp.get_example_data("rlmw_002_short")
        d.write_file(os.path.join(self.server_root, "data", "d.pd"))
        d2 = pp.EyeData.from_file("ssh://server/data/d.pd", cache_dir=self.cache)
        self.assertEqual(len(d2.tx), len(d.tx))


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
        data, config = io.load_study_local(self.study)
        self.assertEqual(data["s1"], ("data s1", "control"))
        self.assertEqual(self.server.calls, [])

    def test_ssh(self):
        data, config = io.load_study_local("ssh://server/study/", cache_dir=self.cache)
        self.assertEqual(data["s1"], ("data s1", "control"))
        self.assertEqual(data["s2"], ("data s2", "patient"))
        self.assertTrue(os.path.exists(os.path.join(self.cache, "server", "study", "s2", "data.txt")))

    def test_ssh_subset(self):
        data, _ = io.load_study_local("ssh://server/study", subjects=["s2"], cache_dir=self.cache)
        self.assertEqual(list(data), ["s2"])
        self.assertFalse(os.path.exists(os.path.join(self.cache, "server", "study", "s1")))

    def test_ssh_autodelete(self):
        data, _ = io.load_study_local("ssh://server/study/", cache_dir=self.cache, autodelete=True)
        self.assertEqual(data["s1"], ("data s1", "control"))
        self.assertFalse(os.path.exists(os.path.join(self.cache, "server", "study", "s1", "data.txt")))

    def test_ssh_temp_dir_removed(self):
        with patch("tempfile.mkdtemp", return_value=os.path.join(self.cache, "tmp")):
            os.makedirs(os.path.join(self.cache, "tmp"))
            data, _ = io.load_study_local("ssh://server/study/", autodelete=True)
        self.assertEqual(len(data), 2)
        self.assertFalse(os.path.exists(os.path.join(self.cache, "tmp")))


if __name__ == "__main__":
    unittest.main()
