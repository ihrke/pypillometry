"""`import pypillometry` loads no heavy packages (they are imported where used)."""
import subprocess
import sys

HEAVY = ("pandas", "matplotlib", "scipy", "mpl_toolkits", "h5py", "requests", "tqdm")


def test_import_loads_no_heavy_packages():
    code = ("import sys, pypillometry; "
            "print(' '.join(sorted({m.split('.')[0] for m in sys.modules})))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    assert [m for m in HEAVY if m in out] == []
