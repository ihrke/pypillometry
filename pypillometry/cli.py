"""
Command-line programs of pypillometry.

- ``pypilview``: open the interactive viewer for an Eyelink EDF file or a
  saved eyedata object
- ``edf2pypil``: read an Eyelink EDF file and save it as a pickle file
- ``pypilinfo``: print a summary or a quality report of an Eyelink EDF file
  or a saved eyedata object
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
    # pickle protocol 2 and higher, zstd- or gzip-compressed pickle
    if head.startswith((b"\x80", b"\x28\xb5\x2f\xfd", b"\x1f\x8b")):
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


def _read_file(source, keep_triggers=False):
    """
    Read an EDF or pickle file (local file or URL).

    Returns the object, the format ("edf" or "pickle") and the file size in
    bytes. Raises ValueError if the file is neither an EDF nor a pickle file.
    """
    import pypillometry as pp
    with pp.local_copy(source) as local:
        fmt = _detect_format(local)
        size = os.path.getsize(local)
        if fmt == "edf":
            d = pp.EyeData.from_eyelink(
                local, remove_eyelink_triggers=not keep_triggers)
        elif fmt == "pickle":
            d = pp.read_pickle(local)
        else:
            raise ValueError(f"{source} is neither an EDF file nor a pickle file")
    return d, fmt, size


def _format_size(nbytes):
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024


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

    try:
        d, fmt, _ = _read_file(args.file, keep_triggers=args.keep_triggers)
    except ValueError as e:
        parser.error(str(e))
    if fmt == "pickle" and not isinstance(d, pp.GenericEyeData):
        parser.error(f"{args.file} does not contain an eyedata object "
                     f"(found {type(d).__name__})")

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


def _pypilinfo_parser():
    parser = argparse.ArgumentParser(
        prog="pypilinfo",
        description="Print a summary of Eyelink EDF files or eyedata objects "
                    "saved with pp.write_pickle() (e.g., .pkl or .pypil). The "
                    "format is detected from the file content. Reading EDF "
                    "files is slow (about a minute for 90 MB); convert them "
                    "with edf2pypil first if you need them more than once.")
    parser.add_argument("files", nargs="+", metavar="file",
                        help="local file, HTTP(S) URL or SSH URL "
                             "(ssh://server/folder/file.pkl)")
    parser.add_argument("--keep-triggers", action="store_true",
                        help="EDF only: keep Eyelink triggers (calibration, "
                             "validation etc.) in the events")
    parser.add_argument("-Q", "--quality", action="store_true",
                        help="print a data-quality report instead of the summary")
    parser.add_argument("--html", action="store_true",
                        help="write the quality reports of all files to an HTML "
                             "file and open it in the browser")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="with --html: name of the HTML file (default: a "
                             "temporary file)")
    parser.add_argument("--no-open", action="store_true",
                        help="with --html: do not open the browser")
    _add_verbosity(parser)
    return parser


def _short_names(sources):
    """File names without directories if they are unique, else `sources` as given."""
    names = [os.path.basename(src.rstrip("/")) or src for src in sources]
    return names if len(set(names)) == len(names) else list(sources)


def _write_html_report(fname, reports):
    """
    Write `reports` (list of (source, header, QualityReport)) to an HTML page.

    For several reports, the page starts with a summary table that links to
    the individual reports.
    """
    from .quality import _html_page
    sources = [src for src, _, _ in reports]
    page = _html_page([(header, r) for _, header, r in reports],
                      names=_short_names(sources), tooltips=sources)
    with open(fname, "w", encoding="utf-8") as f:
        f.write(page)


def pypilinfo_main(argv=None):
    """
    Entry point of the ``pypilinfo`` command.

    Reads each file like ``pypilview`` (EDF or pickle, detected from the
    file content) and prints its format, size and a summary of the stored
    object (its :func:`repr`). With ``--quality``, the data-quality report
    (:meth:`~pypillometry.GenericEyeData.quality_report`) is printed
    instead; for several files, a summary table comes first. With ``--html``, the quality reports of all files are written
    to an HTML file (``-o``, default: a temporary file), which is opened in
    the browser (unless ``--no-open``); for several files, the page starts
    with a summary table that links to the individual reports.
    Files that cannot be read are reported and skipped. Only open pickle files you trust: unpickling can run arbitrary
    code.

    Parameters
    ----------
    argv : list of str, optional
        command-line arguments (without the program name); if None
        (default), ``sys.argv[1:]`` is used

    Returns
    -------
    int
        exit status (0 if all files were read, 1 otherwise)
    """
    import sys
    parser = _pypilinfo_parser()
    args = parser.parse_args(argv)
    _set_verbosity(args)

    if not args.html and (args.output or args.no_open):
        parser.error("-o/--output and --no-open need --html")

    status = 0
    reports = []
    printed = False
    for source in args.files:
        if "://" not in source and not os.path.isfile(source):
            print(f"pypilinfo: file not found: {source}", file=sys.stderr)
            status = 1
            continue
        try:
            d, fmt, size = _read_file(source, keep_triggers=args.keep_triggers)
        except Exception as e:
            print(f"pypilinfo: {source}: {e}", file=sys.stderr)
            status = 1
            continue
        header = f"{source} ({fmt.upper()}, {_format_size(size)})"
        if not (args.quality or args.html):
            if printed:
                print()
            print(header)
            print(repr(d).rstrip("\n"))
            printed = True
            continue
        if not hasattr(d, "quality_report"):
            print(f"pypilinfo: {source}: no quality report for "
                  f"{type(d).__name__} objects", file=sys.stderr)
            status = 1
            continue
        reports.append((source, header, d.quality_report()))

    if args.quality and reports:
        if len(reports) > 1:
            from .quality import _summary_text
            print(_summary_text(_short_names([src for src, _, _ in reports]),
                                [r for _, _, r in reports]))
        for i, (source, header, report) in enumerate(reports):
            if i > 0 or len(reports) > 1:
                print()
            print(header)
            print(report)

    if args.html and reports:
        fname = args.output
        if not fname:
            import tempfile
            fd, fname = tempfile.mkstemp(prefix="pypilinfo_", suffix=".html")
            os.close(fd)
        _write_html_report(fname, reports)
        print(f"Wrote {fname}")
        if not args.no_open:
            import pathlib
            import webbrowser
            webbrowser.open(pathlib.Path(fname).resolve().as_uri())
    return status
