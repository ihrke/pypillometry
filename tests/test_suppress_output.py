"""suppress_all_output() must also hide buffered output of C libraries."""
import subprocess
import sys
import unittest

CODE = """
import ctypes, sys
from pypillometry.convenience import suppress_all_output
print("before", flush=False)
with suppress_all_output():
    ctypes.CDLL(None).printf(b"from C\\n")  # buffered: stdout is a pipe
    print("from Python")
print("after")
"""


@unittest.skipIf(sys.platform == "win32", "uses printf from libc")
class TestSuppressAllOutput(unittest.TestCase):
    def test_c_and_python_output(self):
        r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "before\nafter\n")


if __name__ == "__main__":
    unittest.main()
