"""
Command-line programs of pypillometry.

- ``pypilview``: open the interactive viewer for an Eyelink EDF file
"""
import argparse
import os

# entry points of the command-line programs, not exported at the top level
__all__ = []


def _pypilview_parser():
    parser = argparse.ArgumentParser(
        prog="pypilview",
        description="Open the pypillometry viewer for an Eyelink EDF file. "
                    "Press H in the viewer for the keyboard controls.")
    parser.add_argument("file",
                        help="EDF file, HTTP(S) URL or SSH URL "
                             "(ssh://server/folder/file.edf)")
    parser.add_argument("--eyes", nargs="+", metavar="EYE",
                        help="eyes to show (e.g., left right); default: all")
    parser.add_argument("--variables", nargs="+", metavar="VAR",
                        choices=["pupil", "x", "y"],
                        help="variables to show (pupil, x, y); default: all")
    parser.add_argument("--keep-triggers", action="store_true",
                        help="keep Eyelink triggers (calibration, validation "
                             "etc.) in the events")
    return parser


def pypilview_main(argv=None):
    """
    Entry point of the ``pypilview`` command.

    Reads an EDF file with :meth:`~pypillometry.EyeData.from_eyelink` and
    opens it with :func:`~pypillometry.view`.

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

    d = pp.EyeData.from_eyelink(args.file,
                                remove_eyelink_triggers=not args.keep_triggers)
    pp.view(d, eyes=args.eyes, variables=args.variables)
    return 0
