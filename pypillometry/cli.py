"""
Command-line programs of pypillometry.

- ``pypilview``: open the interactive viewer for an Eyelink EDF file or a
  saved eyedata object
- ``edf2pypil``: read an Eyelink EDF file and save it as a pickle file
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


def _add_verbosity(parser):
    """Add the -q/--quiet and -v/--verbose options to `parser`."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-q", "--quiet", action="store_true",
                       help="show only warnings and errors (no progress bars)")
    group.add_argument("-v", "--verbose", action="store_true",
                       help="show debug messages (including the output of eyelinkio)")


def _set_verbosity(args):
    """Set the log level (and progress bars) from the -q/-v options."""
    from .logging import logging_set_level
    if args.quiet:
        logging_set_level("WARNING")
        os.environ["TQDM_DISABLE"] = "1"  # read by tqdm when a bar is created
    elif args.verbose:
        logging_set_level("DEBUG")
    else:
        logging_set_level("INFO")


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
    _add_verbosity(parser)
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
    _set_verbosity(args)

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


def _default_output(source):
    """``<filebase>.pkl`` for a local file or URL (URLs: in the current directory)."""
    if "://" in source:
        from urllib.parse import urlsplit, unquote
        source = os.path.basename(unquote(urlsplit(source).path).rstrip("/"))
    return os.path.splitext(source)[0] + ".pkl"


def _edf2pypil_parser():
    parser = argparse.ArgumentParser(
        prog="edf2pypil",
        description="Read an Eyelink EDF file and save it as an EyeData object "
                    "in a pickle file (read it with pp.read_pickle() or open "
                    "it with pypilview).")
    parser.add_argument("file",
                        help="EDF file, HTTP(S) URL or SSH URL "
                             "(ssh://server/folder/file.edf)")
    parser.add_argument("-o", "--output",
                        help="output file; default: <filebase>.pkl (next to "
                             "the EDF file, or in the current directory for URLs)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="overwrite the output file if it exists")
    parser.add_argument("--keep-triggers", action="store_true",
                        help="keep Eyelink triggers (calibration, validation "
                             "etc.) in the events")
    _add_verbosity(parser)
    return parser


def edf2pypil_main(argv=None):
    """
    Entry point of the ``edf2pypil`` command.

    Reads an EDF file with :meth:`~pypillometry.EyeData.from_eyelink` and
    saves the object with :func:`~pypillometry.write_pickle`.

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
    parser = _edf2pypil_parser()
    args = parser.parse_args(argv)
    _set_verbosity(args)

    is_url = "://" in args.file
    if not is_url and not os.path.isfile(args.file):
        parser.error(f"file not found: {args.file}")
    output = args.output or _default_output(args.file)
    if os.path.exists(output) and not args.force:
        parser.error(f"output file exists: {output} (use --force to overwrite)")

    # import here so that `edf2pypil --help` is fast
    import pypillometry as pp

    with pp.local_copy(args.file) as local:
        if _detect_format(local) != "edf":
            parser.error(f"{args.file} is not an EDF file")
        d = pp.EyeData.from_eyelink(
            local, remove_eyelink_triggers=not args.keep_triggers)
    pp.write_pickle(d, output)
    if not args.quiet:
        print(f"Saved {output}")
    return 0
