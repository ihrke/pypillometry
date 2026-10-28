"""Tests for the command-line programs (no viewer window is opened)."""
import contextlib
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pypillometry as pp
from pypillometry.cli import pypilview_main

EDF = os.path.join(os.path.dirname(pp.__file__), "data", "test.edf")


class TestPypilview(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, argv):
        with patch.object(pp.EyeData, "from_eyelink", return_value="data") as read, \
             patch.object(pp, "view") as view:
            status = pypilview_main(argv)
        return status, read, view

    def assert_exits(self, argv):
        with self.assertRaises(SystemExit) as cm, patch("sys.stderr"), \
             patch.object(pp, "view") as view:
            pypilview_main(argv)
        self.assertEqual(cm.exception.code, 2)
        view.assert_not_called()

    def test_defaults(self):
        status, read, view = self.run_main([EDF])
        self.assertEqual(status, 0)
        read.assert_called_once_with(EDF, remove_eyelink_triggers=True)
        view.assert_called_once_with("data", eyes=None, variables=None)

    def test_options(self):
        _, read, view = self.run_main([EDF, "--eyes", "left", "--variables", "pupil", "x",
                                       "--keep-triggers"])
        read.assert_called_once_with(EDF, remove_eyelink_triggers=False)
        view.assert_called_once_with("data", eyes=["left"], variables=["pupil", "x"])

    def test_edf_detected_by_content(self):
        fname = os.path.join(self.tmp, "recording.dat")
        shutil.copy(EDF, fname)
        _, read, _ = self.run_main([fname])
        read.assert_called_once_with(fname, remove_eyelink_triggers=True)

    def test_url_is_downloaded(self):
        @contextlib.contextmanager
        def local_copy(source):
            self.assertEqual(source, "ssh://server/data/test.edf")
            yield EDF
        with patch.object(pp, "local_copy", side_effect=local_copy):
            _, read, _ = self.run_main(["ssh://server/data/test.edf"])
        read.assert_called_once_with(EDF, remove_eyelink_triggers=True)

    def test_pickle_any_extension(self):
        d = pp.PupilData(left_pupil=np.ones(100), sampling_rate=100, name="saved")
        for ext in (".pkl", ".pypil", ""):
            fname = os.path.join(self.tmp, "data" + ext)
            pp.write_pickle(d, fname)
            with patch.object(pp, "view") as view:
                pypilview_main([fname, "--eyes", "left"])
            shown = view.call_args.args[0]
            self.assertIsInstance(shown, pp.PupilData)
            self.assertEqual(shown.name, "saved")
            self.assertEqual(view.call_args.kwargs["eyes"], ["left"])

    def test_pickle_without_eyedata(self):
        fname = os.path.join(self.tmp, "other.pkl")
        pp.write_pickle({"a": 1}, fname)
        self.assert_exits([fname])

    def test_unknown_format(self):
        fname = os.path.join(self.tmp, "data.edf")
        with open(fname, "w") as f:
            f.write("time,pupil\n")
        self.assert_exits([fname])

    def test_missing_file(self):
        self.assert_exits(["does_not_exist.edf"])

    def test_invalid_variable(self):
        self.assert_exits([EDF, "--variables", "foo"])

    def test_reads_edf(self):
        # the real file is read; only the viewer is mocked
        with patch.object(pp, "view") as view:
            pypilview_main([EDF])
        d = view.call_args.args[0]
        self.assertIsInstance(d, pp.EyeData)
        self.assertGreater(len(d), 0)


if __name__ == "__main__":
    unittest.main()
