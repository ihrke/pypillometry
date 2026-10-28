"""Tests for the command-line programs (no viewer window is opened)."""
import contextlib
import os
import shutil
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np

import pypillometry as pp
from pypillometry.cli import pypilview_main, edf2pypil_main, _default_output

EDF = os.path.join(os.path.dirname(pp.__file__), "data", "test.edf")


class CLITestCase(unittest.TestCase):
    """The programs set the log level (and TQDM_DISABLE): restore both after each test."""

    def setUp(self):
        level = pp.logging_get_level()
        self.addCleanup(pp.logging_set_level, level)
        p = patch.dict(os.environ)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("TQDM_DISABLE", None)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)


class TestPypilview(CLITestCase):
    def run_main(self, argv):
        with patch.object(pp.EyeData, "from_eyelink", return_value="data") as read, \
             patch.object(pp, "view") as view:
            status = pypilview_main(argv)
        return status, read, view

    def assert_exits(self, argv):
        with self.assertRaises(SystemExit) as cm, patch("sys.stderr", new_callable=StringIO), \
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


class TestEdf2pypil(CLITestCase):
    def setUp(self):
        super().setUp()
        self.edf = os.path.join(self.tmp, "rec.edf")
        shutil.copy(EDF, self.edf)

    def run_main(self, argv):
        with patch("sys.stdout", new_callable=StringIO):
            return edf2pypil_main(argv)

    def assert_exits(self, argv):
        with self.assertRaises(SystemExit) as cm, patch("sys.stderr", new_callable=StringIO):
            edf2pypil_main(argv)
        self.assertEqual(cm.exception.code, 2)

    def test_default_output(self):
        self.assertEqual(self.run_main([self.edf]), 0)
        d = pp.read_pickle(os.path.join(self.tmp, "rec.pkl"))
        self.assertIsInstance(d, pp.EyeData)
        self.assertEqual(len(d), len(pp.EyeData.from_eyelink(EDF)))

    def test_output_option(self):
        out = os.path.join(self.tmp, "out.pypil")
        self.run_main([self.edf, "-o", out])
        self.assertIsInstance(pp.read_pickle(out), pp.EyeData)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "rec.pkl")))

    def test_keep_triggers(self):
        with patch.object(pp.EyeData, "from_eyelink", return_value={"a": 1}) as read:
            self.run_main([self.edf, "--keep-triggers"])
        read.assert_called_once_with(self.edf, remove_eyelink_triggers=False)

    def test_no_overwrite_without_force(self):
        out = os.path.join(self.tmp, "rec.pkl")
        with open(out, "w") as f:
            f.write("old")
        self.assert_exits([self.edf])
        with open(out) as f:
            self.assertEqual(f.read(), "old")
        self.run_main([self.edf, "--force"])
        self.assertIsInstance(pp.read_pickle(out), pp.EyeData)

    def test_not_an_edf(self):
        fname = os.path.join(self.tmp, "data.edf")
        pp.write_pickle({"a": 1}, fname)
        self.assert_exits([fname])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "data.pkl")))

    def test_missing_file(self):
        self.assert_exits([os.path.join(self.tmp, "missing.edf")])

    def test_default_output_names(self):
        self.assertEqual(_default_output("data/rec.edf"), "data/rec.pkl")
        self.assertEqual(_default_output("rec.EDF"), "rec.pkl")
        self.assertEqual(_default_output("ssh://server/data/rec.edf"), "rec.pkl")
        self.assertEqual(_default_output("https://example.com/a/rec%201.edf"), "rec 1.pkl")


class TestVerbosity(CLITestCase):
    def setUp(self):
        super().setUp()
        self.edf = os.path.join(self.tmp, "rec.edf")
        shutil.copy(EDF, self.edf)

    def run_both(self, opts):
        """Run both programs with `opts`; return (log level, TQDM_DISABLE, stdout) of each."""
        out = []
        for main, argv in [(pypilview_main, [self.edf]), (edf2pypil_main, [self.edf, "-f"])]:
            stdout = StringIO()
            with patch.object(pp.EyeData, "from_eyelink", return_value={}), \
                 patch.object(pp, "view"), patch("sys.stdout", stdout):
                main(argv + opts)
            out.append((pp.logging_get_level(), os.environ.get("TQDM_DISABLE"), stdout.getvalue()))
            os.environ.pop("TQDM_DISABLE", None)
        return out

    def test_default(self):
        for level, bars, _ in self.run_both([]):
            self.assertEqual(level, "INFO")
            self.assertIsNone(bars)
        self.assertIn("Saved", self.run_both([])[1][2])

    def test_quiet(self):
        for opt in ("-q", "--quiet"):
            for level, bars, stdout in self.run_both([opt]):
                self.assertEqual(level, "WARNING")
                self.assertEqual(bars, "1")
                self.assertEqual(stdout, "")

    def test_verbose(self):
        for opt in ("-v", "--verbose"):
            for level, bars, _ in self.run_both([opt]):
                self.assertEqual(level, "DEBUG")
                self.assertIsNone(bars)

    def test_quiet_and_verbose_exclusive(self):
        for main in (pypilview_main, edf2pypil_main):
            with self.assertRaises(SystemExit), patch("sys.stderr", new_callable=StringIO):
                main([self.edf, "-q", "-v"])


if __name__ == "__main__":
    unittest.main()
