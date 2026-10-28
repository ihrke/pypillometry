"""Tests for the command-line programs (no viewer window is opened)."""
import os
import unittest
from unittest.mock import patch

import pypillometry as pp
from pypillometry.cli import pypilview_main

EDF = os.path.join(os.path.dirname(pp.__file__), "data", "test.edf")


class TestPypilview(unittest.TestCase):
    def run_main(self, argv):
        with patch.object(pp.EyeData, "from_eyelink", return_value="data") as read, \
             patch.object(pp, "view") as view:
            status = pypilview_main(argv)
        return status, read, view

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

    def test_url_is_not_checked_locally(self):
        _, read, _ = self.run_main(["ssh://server/data/test.edf"])
        read.assert_called_once()

    def test_missing_file(self):
        with self.assertRaises(SystemExit) as cm, patch("sys.stderr"):
            pypilview_main(["does_not_exist.edf"])
        self.assertEqual(cm.exception.code, 2)

    def test_invalid_variable(self):
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            pypilview_main([EDF, "--variables", "foo"])

    def test_reads_edf(self):
        # the real file is read; only the viewer is mocked
        with patch.object(pp, "view") as view:
            pypilview_main([EDF])
        d = view.call_args.args[0]
        self.assertIsInstance(d, pp.EyeData)
        self.assertGreater(len(d), 0)


if __name__ == "__main__":
    unittest.main()
