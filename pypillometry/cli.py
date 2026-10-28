"""
Command-line programs of pypillometry.

- ``pypilview``: open the interactive viewer for an Eyelink EDF file or a
  saved eyedata object
"""
import argparse
import os

# entry points of the command-line programs, not exported at the top level
__all__ = []


def _detect_format(fname):
    """Return "edf" or "pickle" based on the first bytes of a local file, else None."""
    with open(fname, "rb") as f:
        head = f.read(16)
    if head.startswith(b"SR_RESEARCH"):
        return "edf"
    if head.startswith(b"\x80"):  # pickle protocol 2 and higher
        return "pickle"
    return None


def _pypilview_parser():
    parser = argparse.ArgumentParser(
        prog="pypilview",
        description="Open the pypillometry viewer for an Eyelink EDF file or "
                    "an eyedata object saved with pp.write_pickle() (e.g., "
                    ".pkl or .pypil). The format is detected from the file "
                    "content. Press H in the viewer for the keyboard controls.")
    parser.add_argument("file",
                        help="local file, HTTP(S) URL or SSH URL "
                             "(ssh://server/folder/file.edf)")
    parser.add_argument("--eyes", nargs="+", metavar="EYE",
                        help="eyes to show (e.g., left right); default: all")
    parser.add_argument("--variables", nargs="+", metavar="VAR",
                        choices=["pupil", "x", "y"],
                        help="variables to show (pupil, x, y); default: all")
    parser.add_argument("--keep-triggers", action="store_true",
                        help="EDF only: keep Eyelink triggers (calibration, "
                             "validation etc.) in the events")
    return parser


def pypilview_main(argv=None):
    """
    Entry point of the ``pypilview`` command.

    Reads an EDF file with :meth:`~pypillometry.EyeData.from_eyelink` or a
    pickled eyedata object with :func:`~pypillometry.read_pickle` (the
    format is detected from the first bytes of the file, so the extension
    does not matter) and opens it with :func:`~pypillometry.view`. URLs are
    downloaded to a temporary directory first. Only open pickle files you
    trust: unpickling can run arbitrary code.

    Parameters
    ----------
    argv : list of str, optional
        command-line arguments (without the program name); if None
        (default), ``sys.argv[1:]`` is used

    Returns
    -------
    int
        exit status (0 on success)
    """
    parser = _pypilview_parser()
    args = parser.parse_args(argv)

    is_url = "://" in args.file
    if not is_url and not os.path.isfile(args.file):
        parser.error(f"file not found: {args.file}")

    # import here so that `pypilview --help` is fast
    import pypillometry as pp

    with pp.local_copy(args.file) as local:
        fmt = _detect_format(local)
        if fmt == "edf":
            d = pp.EyeData.from_eyelink(
                local, remove_eyelink_triggers=not args.keep_triggers)
        elif fmt == "pickle":
            d = pp.read_pickle(local)
            if not isinstance(d, pp.GenericEyeData):
                parser.error(f"{args.file} does not contain an eyedata object "
                             f"(found {type(d).__name__})")
        else:
            parser.error(f"{args.file} is neither an EDF file nor a pickle file")

    pp.view(d, eyes=args.eyes, variables=args.variables)
    return 0
