"""`import pypillometry` must not load slow modules that are rarely needed."""
import subprocess
import sys
import unittest


def loaded_after_import(modules):
    """Return the subset of `modules` loaded by `import pypillometry` (fresh process)."""
    code = ("import sys, pypillometry; "
            f"print(' '.join(m for m in {list(modules)!r} if m in sys.modules))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout.split()


class TestImport(unittest.TestCase):
    def test_no_cmdstanpy(self):
        # cmdstanpy is only needed for the Stan baseline model
        self.assertEqual(loaded_after_import(["cmdstanpy", "readline"]), [])


if __name__ == "__main__":
    unittest.main()
