"""
Command-line programs of pypillometry.

- ``pypilview``: open the interactive viewer for an Eyelink EDF file or a
  saved eyedata object
- ``edf2pypil``: read an Eyelink EDF file and save it as a pickle (or HDF5)
  file
- ``pypilinfo``: print a summary or a quality report of an Eyelink EDF file
  or a saved eyedata object
- ``pypil-upgrade``: convert files saved with an older version of
  pypillometry to the current file format
"""
import argparse
import os
import sys

# entry points of the command-line programs, not exported at the top level
__all__ = []


def _detect_format(fname):
    """Return "edf", "pickle" or "hdf5" based on the first bytes of a local file, else None."""
    with open(fname, "rb") as f:
        head = f.read(16)
    if head.startswith(b"SR_RESEARCH"):
        return "edf"
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "hdf5"
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
    Read an EDF, pickle or HDF5 file (local file or URL).

    Returns the object, the format ("edf", "pickle" or "hdf5") and the file
    size in bytes. Raises ValueError if the file has none of these formats.
    """
    import pypillometry as pp
    with pp.local_copy(source) as local:
        fmt = _detect_format(local)
        size = os.path.getsize(local)
        if fmt == "edf":
            d = pp.EyeData.from_eyelink(
                local, remove_eyelink_triggers=not keep_triggers)
        elif fmt in ("pickle", "hdf5"):
            d = pp.read_pickle(local)
        else:
            raise ValueError(f"{source} is neither an EDF file nor a pickle "
                             f"or HDF5 file")
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
                    ".pkl, .pypil or .h5). The format is detected from the file "
                    "content. Press H in the viewer for the keyboard controls.")
    # `file` is optional for argparse so that it may follow --eyes/--variables
    # (which take several values), see _file_after_list_option
    parser.add_argument("file", nargs="?",
                        help="local file, HTTP(S) URL or SSH URL "
                             "(ssh://server/folder/file.edf)")
    parser.add_argument("--eyes", nargs="+", metavar="EYE",
                        help="eyes to show (e.g., left right); default: all")
    parser.add_argument("--variables", nargs="+", metavar="VAR",
                        help="variables to show (pupil, x, y); default: all")
    parser.add_argument("--keep-triggers", action="store_true",
                        help="EDF only: keep Eyelink triggers (calibration, "
                             "validation etc.) in the events")
    _add_verbosity(parser)
    return parser


_VIEW_VARIABLES = ("pupil", "x", "y")


def _file_after_list_option(parser, args, argv):
    """pypilview: take the file from the end of --eyes/--variables if it was
    given after them (``pypilview --eyes right FILE``), check the variables."""
    if args.file is None:
        # (argparse also accepts abbreviations, e.g. --eye)
        given = [(i, opt) for i, a in enumerate(argv)
                 for opt in ("eyes", "variables")
                 if len(a) > 3 and f"--{opt}".startswith(a)]
        values = getattr(args, max(given)[1]) if given else None
        if not values:
            parser.error("the following arguments are required: file")
        args.file = values.pop()
        if not values:
            parser.error(f"--{max(given)[1]} needs at least one value before the file")
    bad = [v for v in args.variables or [] if v not in _VIEW_VARIABLES]
    if bad:
        parser.error(f"argument --variables: invalid choice: {', '.join(map(repr, bad))} "
                     f"(choose from {', '.join(_VIEW_VARIABLES)})")


def pypilview_main(argv=None):
    """
    Entry point of the ``pypilview`` command.

    Reads an EDF file with :meth:`~pypillometry.EyeData.from_eyelink` or an
    eyedata object saved in a pickle or HDF5 file with
    :func:`~pypillometry.read_pickle` (the format is detected from the first
    bytes of the file, so the extension does not matter) and opens it with :func:`~pypillometry.view`. URLs are
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
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    _file_after_list_option(parser, args, argv)
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
    if fmt != "edf" and not isinstance(d, pp.GenericEyeData):
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
                    "in a pickle file, or an HDF5 file if the output ends in "
                    ".h5 or .hdf5 (read it with pp.read_pickle() or open "
                    "it with pypilview).")
    parser.add_argument("file",
                        help="EDF file, HTTP(S) URL or SSH URL "
                             "(ssh://server/folder/file.edf)")
    parser.add_argument("-o", "--output",
                        help="output file (.h5 or .hdf5: HDF5, otherwise pickle); "
                             "default: <filebase>.pkl (next to the EDF file, or "
                             "in the current directory for URLs)")
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
    saves the object with :func:`~pypillometry.write_pickle` (as HDF5 if the
    output file ends in ``.h5`` or ``.hdf5``).

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
                    "saved with pp.write_pickle() (e.g., .pkl, .pypil or .h5). The "
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
    parser.add_argument("-d", "--descriptions", action="store_true",
                        help="with -Q: explain each quality measure and its "
                             "thresholds (in the HTML report, hover over a measure)")
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

    Reads each file like ``pypilview`` (EDF, pickle or HDF5, detected from
    the file content) and prints its format, size and a summary of the stored
    object (its :func:`repr`). With ``--quality``, the data-quality report
    (:meth:`~pypillometry.GenericEyeData.quality_report`) is printed
    instead (with ``--descriptions``, each measure is explained); for several
    files, a summary table comes first. With ``--html``, the quality reports of all files are written
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
    if args.descriptions and not args.quality:
        parser.error("-d/--descriptions needs -Q/--quality")

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
            print(report.to_text(descriptions=args.descriptions))

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


def _upgrade_parser():
    parser = argparse.ArgumentParser(
        prog="pypil-upgrade",
        description="Convert pickle and HDF5 files saved with an older version of "
                    "pypillometry (e.g., with pp.write_pickle() or write_file()) to the "
                    "current file format. Old files are also upgraded whenever they are "
                    "read; this converts them once. Each file keeps its format (pickle "
                    "or HDF5, detected from the content); other files are left alone. "
                    "Only convert pickle files you trust: unpickling can run arbitrary "
                    "code.")
    parser.add_argument("paths", nargs="+", metavar="path",
                        help="files or folders (the files directly in a folder; with -r, "
                             "also those in its subfolders)")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("-o", "--output", metavar="PATH",
                        help="write the converted files here: a file name for a single "
                             "input file, otherwise a folder (created if needed; files "
                             "of an input folder keep their paths relative to it); "
                             "files that are up to date are not copied")
    target.add_argument("-i", "--in-place", action="store_true",
                        help="replace the old files with the converted ones")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="only report what would be converted (no -o/-i needed)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="with -o: overwrite existing output files")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="also convert the files in subfolders of the given folders")
    _add_verbosity(parser)
    return parser


def _upgrade_sources(paths, recursive):
    """List of (file, path relative to the given folder or None) for `paths`."""
    out = []
    for path in paths:
        if os.path.isfile(path):
            out.append((path, None))
            continue
        for root, dirs, files in os.walk(path):
            dirs.sort()
            if not recursive:
                dirs.clear()
            for name in sorted(files):
                fname = os.path.join(root, name)
                out.append((fname, os.path.relpath(fname, path)))
    return out


def _upgrade_targets(sources, output, single):
    """Output file for each source (None: in place)."""
    if output is None:
        return [None] * len(sources)
    if single and not (os.path.isdir(output) or output.endswith(os.sep)):
        return [output]
    return [os.path.join(output, rel or os.path.basename(src)) for src, rel in sources]


def _load_versions(fname):
    """Load `fname` (pickle or HDF5); return the object and the format versions
    of the pypillometry objects in it (before they were upgraded)."""
    from . import state
    from .io import read_pickle
    with state._recording_versions() as seen:
        obj = read_pickle(fname)
    return obj, [version for _, version in seen]


def _is_pickle(fname):
    """True if `fname` (detected as "pickle") starts with the pickle protocol
    marker after decompression (e.g., not some other gzip file)."""
    from .io import _open_maybe_compressed
    try:
        with _open_maybe_compressed(fname) as f:
            return f.read(1) == b"\x80"
    except Exception:
        return False


def _write_like(obj, fmt, source, target):
    """Write `obj` to `target` in the format of `source` (`fmt`: "pickle" or
    "hdf5"; pickles compressed if `source` was); via a temporary file."""
    from . import hdf5, state
    from .io import write_pickle
    folder = os.path.dirname(os.path.abspath(target))
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, f".{os.path.basename(target)}.pypil-upgrade.tmp")
    try:
        if fmt == "hdf5":
            hdf5.write_state(state.to_state(obj), tmp)
        else:
            with open(source, "rb") as f:
                plain = f.read(1) == b"\x80"
            write_pickle(obj, tmp, compress=not plain)
        if os.path.exists(target):
            import shutil
            shutil.copymode(target, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def pypil_upgrade_main(argv=None):
    """
    Entry point of the ``pypil-upgrade`` command.

    Reads each pickle or HDF5 file (the format is detected from the file
    content) with :func:`~pypillometry.read_pickle`, which upgrades the
    objects saved with an older version of pypillometry, and writes it again
    in the same format: in place (``-i``) or to ``-o``. Files that are
    already in the current format are not written; files that are neither
    pickle nor HDF5 files, or contain no pypillometry objects, are left
    alone. A line for each file (unless ``-q``) and a summary are printed;
    with ``-n``, nothing is written.

    Parameters
    ----------
    argv : list of str, optional
        command-line arguments (without the program name); if None
        (default), ``sys.argv[1:]`` is used

    Returns
    -------
    int
        exit status (0 if no file failed, 1 otherwise)
    """
    import sys
    parser = _upgrade_parser()
    args = parser.parse_args(argv)
    _set_verbosity(args)
    if not (args.dry_run or args.output or args.in_place):
        parser.error("give -o/--output or -i/--in-place (or -n/--dry-run)")
    if args.force and not args.output:
        parser.error("-f/--force needs -o/--output")
    missing = [p for p in args.paths if not os.path.exists(p)]
    if missing:
        parser.error(f"not found: {', '.join(missing)}")

    from .state import FORMAT_VERSION
    sources = _upgrade_sources(args.paths, args.recursive)
    single = len(args.paths) == 1 and os.path.isfile(args.paths[0])
    targets = _upgrade_targets(sources, args.output, single)
    if args.output:
        abs_targets = [os.path.abspath(t) for t in targets]
        if len(set(abs_targets)) < len(abs_targets):
            parser.error("several input files would be written to the same output file")
        if set(abs_targets) & {os.path.abspath(src) for src, _ in sources}:
            parser.error("an output file is one of the input files (use -i to convert "
                         "in place)")

    def report(line):
        if not args.quiet:
            print(line)

    counts = {"upgraded": 0, "current": 0, "skipped": 0, "failed": 0}
    for (source, _), target in zip(sources, targets):
        fmt = _detect_format(source)
        if fmt not in ("pickle", "hdf5") or (fmt == "pickle" and not _is_pickle(source)):
            report(f"{source}: skipped (not a pickle or HDF5 file)")
            counts["skipped"] += 1
            continue
        try:
            obj, versions = _load_versions(source)
        except Exception as e:
            print(f"pypil-upgrade: {source}: cannot read: {e}", file=sys.stderr)
            counts["failed"] += 1
            continue
        if not versions:
            report(f"{source}: skipped (no pypillometry objects)")
            counts["skipped"] += 1
            continue
        old = min(versions)
        if old >= FORMAT_VERSION:
            report(f"{source}: up to date (format {FORMAT_VERSION})")
            counts["current"] += 1
            continue
        out = target or os.path.realpath(source)
        change = f"format {old} -> {FORMAT_VERSION} ({type(obj).__name__}, {fmt})"
        if args.dry_run:
            report(f"{source}: would convert, {change}" + (f", to {target}" if target else ""))
            counts["upgraded"] += 1
            continue
        if target and os.path.exists(target) and not args.force:
            print(f"pypil-upgrade: {source}: output file exists: {target} (use -f to "
                  f"overwrite)", file=sys.stderr)
            counts["failed"] += 1
            continue
        try:
            _write_like(obj, fmt, source, out)
        except Exception as e:
            print(f"pypil-upgrade: {source}: cannot write {out}: {e}", file=sys.stderr)
            counts["failed"] += 1
            continue
        report(f"{source}: converted, {change}" + (f", written to {target}" if target else ""))
        counts["upgraded"] += 1

    verb = "would be converted" if args.dry_run else "converted"
    print(f"{len(sources)} file(s): {counts['upgraded']} {verb}, {counts['current']} up to "
          f"date, {counts['skipped']} skipped, {counts['failed']} failed")
    return 1 if counts["failed"] else 0
