"""
io.py
=====

Read/Write data from/to disk.

Reading EDF files is slow: :func:`read_eyelink` (and
:meth:`~pypillometry.GenericEyeData.from_eyelink`) takes about a minute
for a 90 MB EDF file (64 min of binocular data at 1000 Hz), mostly in the
`eyelinkio` package. It is therefore recommended to convert EDF files once
and to work with the converted files, e.g., with the command-line program
``edf2pypil`` (``edf2pypil recording.edf`` writes ``recording.pkl``) or
with :func:`write_pickle`. Reading the pickle file with :func:`read_pickle`
took 0.1-0.25 s for the same recordings (more than 200 times faster; from
a slow disk it can take a few seconds). Pickle files are larger (about
400 MB for a 90 MB EDF file) and should only be read from trusted sources.
"""

try:
   import cPickle as pickle
except:
   import pickle
import os
import requests
from contextlib import contextmanager
from tqdm import tqdm
from typing import Optional
from loguru import logger
from .convenience import is_url, suppress_all_output, requires_package
from .logging import logging_get_level

__all__ = [
    "osf_authenticate", "download", "fetch_url", "local_copy", "write_pickle",
    "read_pickle", "read_eyelink",
]


def osf_authenticate(
    access_token: str,
    validate: bool = True,
    session: Optional[requests.Session] = None,
    timeout: float = 10.0,
) -> requests.Session:
    """
    Authenticate against the OSF API using a personal access token.

    Parameters
    ----------
    access_token : str
        Personal access token generated on OSF.
    validate : bool, optional
        If True (default), verify the token by calling ``/v2/users/me/``.
    session : requests.Session, optional
        Existing session to configure. If None, a new session is created.
    timeout : float, optional
        Timeout (in seconds) for the validation request when ``validate`` is True.

    Returns
    -------
    requests.Session
        Session configured with the OSF bearer token.

    Raises
    ------
    ValueError
        If no access token is provided or validation fails.
    """
    if not access_token or not isinstance(access_token, str):
        raise ValueError("A non-empty OSF access token string is required.")

    token = access_token.strip()
    if not token:
        raise ValueError("Access token cannot be empty or whitespace.")

    session = session or requests.Session()

    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
    )

    if validate:
        try:
            response = session.get("https://api.osf.io/v2/users/me/", timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise ValueError(f"OSF token validation failed (status {status}).") from exc
        except requests.exceptions.RequestException as exc:
            raise ValueError("OSF token validation request failed.") from exc
        else:
            logger.info("OSF token validation succeeded.")

    return session


def _osf_client_url_to_download_url(url: str, session: Optional[requests.Session] = None) -> str:
    """
    Convert an OSF client URL to a download URL.
    Parameters
    ----------
    url : str
        OSF client URL
    session : requests.Session, optional
        requests session to use for the download. If None, a new session is created.
    Returns
    -------
    str
        Download URL
    """
    # is it an OSF client URL?
    if not url.startswith("https://osf.io/"):
        logger.debug(f"URL is not an OSF client URL: {url!r}")
        return url
    
    # Check if it's already a download URL
    if "/download/" in url:
        logger.debug(f"URL is already a download URL: {url!r}")
        return url
    
    # get the project ID
    from urllib.parse import urlsplit
    parts = [segment for segment in urlsplit(url).path.split('/') if segment]
    if not parts:
        logger.error(f"No OSF identifier found in {url!r}")
        return url
    file_id = parts[0]  
    logger.debug(f"File ID: {file_id}")

    # make a get request to https://api.osf.io/v2/files/{file_id}/
    if session is not None:
        response = session.get(f"https://api.osf.io/v2/files/{file_id}/")
    else:
        response = requests.get(f"https://api.osf.io/v2/files/{file_id}/")
    download_url = response.json()["data"]["links"]["move"].replace("/move", "/download")
    logger.debug(f"Download URL: {download_url}")
    return download_url



def download(url: str, fname: str = None, chunk_size: int = 1024, session: Optional[requests.Session] = None) -> str:
    """Download a file from a URL to a local file.

    See https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51.

    Parameters
    ----------
    url : str
        URL of the file to download.
    fname : str, optional
        Local filename to save the file to. If None, create a temporary file.
    chunk_size : int, optional
        Size of the chunks to download the file in. Default is 1024.
    session : requests.Session, optional
        requests session to use for the download. If None, a new session is created.

    Returns
    -------
    str
        Local filename of the downloaded file.
    """
    import tempfile

    # Create temporary file if fname is None
    if fname is None:
        # Extract file extension from URL if possible
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        path = parsed_url.path
        if path and "." in os.path.basename(path):
            suffix = os.path.splitext(path)[1]
        else:
            suffix = ""

        # Create temporary file
        fd, fname = tempfile.mkstemp(suffix=suffix)
        os.close(fd)  # Close the file descriptor, we'll open it again below

    # convert the URL to a download URL if it is an OSF client URL
    url = _osf_client_url_to_download_url(url, session=session)

    own_session = session is None
    session = session or requests.Session()
    try:
        with session.get(url, stream=True) as resp:
            resp.raise_for_status()
            # unknown size (no content-length): progress bar without total
            total = int(resp.headers.get("content-length", 0)) or None
            # fetch_url() downloads to "<name>.part"
            name = os.path.basename(fname)
            if name.endswith(".part"):
                name = name[:-len(".part")]
            with open(fname, "wb") as file, tqdm(
                desc=name,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)
    finally:
        if own_session:
            session.close()

    return fname

def is_ssh_url(source) -> bool:
    """Check if `source` is an SSH URL (``ssh://[user@]host[:port]/path``).

    Parameters
    ----------
    source : str
        string to check

    Returns
    -------
    bool
        True if `source` starts with ``ssh://``
    """
    return isinstance(source, str) and source.startswith("ssh://")


def _parse_ssh_url(url: str):
    """Split an SSH URL into (host, port, remote path).

    `host` may contain a user (``user@host``). The path is absolute
    (``ssh://host/data/f.edf`` -> ``/data/f.edf``) unless it starts with
    ``~/``; then it is relative to the home directory on the server
    (``ssh://host/~/data/f.edf`` -> ``data/f.edf``).
    """
    from urllib.parse import urlsplit, unquote
    parts = urlsplit(url)
    if parts.scheme != "ssh" or not parts.hostname:
        raise ValueError(f"Not a valid SSH URL: '{url}' (expected ssh://host/path)")
    host = parts.hostname
    if parts.username:
        host = f"{parts.username}@{host}"
    path = unquote(parts.path)
    if path in ("", "/"):
        raise ValueError(f"SSH URL '{url}' has no path")
    if path.startswith("/~/"):
        path = path[3:]
    elif path == "/~":
        raise ValueError(f"SSH URL '{url}' has no path")
    return host, parts.port, path


def _cache_path(url: str, cache_dir: str) -> str:
    """Local path in `cache_dir` for a URL: ``cache_dir/<host>/<path>``.

    For SSH URLs with a path relative to the home directory, ``~`` is
    inserted after the host.
    """
    from urllib.parse import urlsplit, unquote
    if is_ssh_url(url):
        host, port, path = _parse_ssh_url(url)
        rel = path.strip("/").split("/")
        if not path.startswith("/"):
            rel = ["~"] + rel  # relative to home on the server
    else:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
        rel = [p for p in unquote(parts.path).split("/") if p] or ["index"]
    host_dir = host if port is None else f"{host}_{port}"
    return os.path.join(cache_dir, host_dir, *rel)


def _ssh_remote_size(host: str, port, path: str) -> Optional[int]:
    """Size in bytes of a file (or folder, if `path` ends with ``/``) on an SSH server.

    Uses ``wc -c`` for files and ``du -sk`` for folders (approximate).
    Returns None if the size cannot be determined.
    """
    import shlex
    import shutil
    import subprocess

    if shutil.which("ssh") is None:
        return None
    is_dir = path.endswith("/")
    qpath = shlex.quote(path.rstrip("/") if is_dir else path)
    remote = f"du -sk {qpath}" if is_dir else f"wc -c < {qpath}"
    cmd = ["ssh", "-o", "BatchMode=yes"]
    if port is not None:
        cmd += ["-p", str(port)]
    cmd += [host, remote]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        size = int(result.stdout.split()[0]) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        size = None
    if size is None:
        logger.debug(f"Could not get the size of '{host}:{path}'")
        return None
    return size * 1024 if is_dir else size


def _local_size(path: str) -> int:
    """Size in bytes of a local file or folder (0 if it does not exist)."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, fnames in os.walk(path):
        for fname in fnames:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:  # file removed while walking
                pass
    return total


def _scp(url: str, target: str):
    """Copy the file (or folder, if `url` ends with ``/``) at an SSH URL to `target`.

    Shows a progress bar that follows the size of `target` while ``scp``
    is running.
    """
    import subprocess
    import shutil
    import threading

    if shutil.which("scp") is None:
        raise RuntimeError("Downloading from SSH URLs requires the 'scp' command")
    host, port, path = _parse_ssh_url(url)
    is_dir = path.endswith("/")
    cmd = ["scp", "-q", "-o", "BatchMode=yes"]
    if port is not None:
        cmd += ["-P", str(port)]
    if is_dir:
        cmd.append("-r")
    cmd += [f"{host}:{path.rstrip('/') if is_dir else path}", target]

    total = _ssh_remote_size(host, port, path)
    # run scp in a thread and follow the size of the local copy
    out = {}
    def run():
        try:
            out["result"] = subprocess.run(cmd, capture_output=True, text=True)
        except BaseException as e:
            out["error"] = e
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    with tqdm(desc=os.path.basename(path.rstrip("/")), total=total,
              unit="iB", unit_scale=True, unit_divisor=1024) as bar:
        while worker.is_alive():
            worker.join(0.2)
            bar.update(_local_size(target) - bar.n)
        # final size (also if scp finished before the first update)
        bar.update(_local_size(target) - bar.n)
    if "error" in out:
        raise out["error"]
    result = out["result"]
    if result.returncode != 0:
        msg = result.stderr.strip()
        raise RuntimeError(
            f"Could not download '{url}' via SSH (scp exit code "
            f"{result.returncode}): {msg}\n"
            "Note: only key-based SSH login is supported (no passwords)."
        )


def fetch_url(
    url: str,
    cache_file: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    session: Optional[requests.Session] = None) -> str:
    """Download an HTTP(S) or SSH URL to a local file, with caching.

    HTTP(S) URLs are downloaded with :func:`download` (OSF client URLs
    such as ``https://osf.io/<id>`` are converted to download URLs).

    SSH URLs have the form ``ssh://[user@]host[:port]/path`` and are
    downloaded with the system's ``scp``, which uses ``~/.ssh/config``
    (host aliases, users, keys). Only key-based login is supported: password
    prompts are disabled (``BatchMode=yes``). The path is absolute on the
    server; use ``ssh://host/~/path`` for a path relative to the home
    directory. An SSH URL ending in ``/`` downloads the whole folder.

    Parameters
    ----------
    url : str
        HTTP(S) or SSH URL
    cache_file : str, optional
        local filename to store the download in
    cache_dir : str, optional
        directory to store the download in (used if `cache_file` is None);
        the file is saved as ``cache_dir/<host>/<path>``. If both are None, a
        new temporary directory is used. The location is logged.
    force_download : bool, optional
        if False (default), an existing local copy is reused; if True, the
        file is downloaded again
    session : requests.Session, optional
        requests session for HTTP(S) downloads (e.g., from
        :func:`osf_authenticate` for private OSF projects)

    Returns
    -------
    str
        local path of the downloaded file (or folder)
    """
    import shutil
    import tempfile

    if not is_url(url):  # also true for SSH URLs
        raise ValueError(f"Not a URL: '{url}'")
    if cache_file is None:
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="pypillometry_")
        cache_file = _cache_path(url, cache_dir)

    if os.path.exists(cache_file) and not force_download:
        logger.info(f"Using cached copy of '{url}': {cache_file}")
        return cache_file

    parent = os.path.dirname(cache_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # download to a temporary name, so that an interrupted download
    # is not used as a cached copy later
    part = cache_file + ".part"
    if os.path.isdir(part):
        shutil.rmtree(part)
    elif os.path.exists(part):
        os.remove(part)

    logger.info(f"Downloading '{url}' to {cache_file}")
    if is_ssh_url(url):
        _scp(url, part)
    else:
        download(url, part, session=session)
    if os.path.isdir(cache_file):
        shutil.rmtree(cache_file)
    os.replace(part, cache_file)
    return cache_file


@contextmanager
def local_copy(
    source: str,
    cache_file: Optional[str] = None,
    cache_dir: Optional[str] = None,
    autodelete: Optional[bool] = None,
    force_download: bool = False,
    session: Optional[requests.Session] = None):
    """Context manager that gives a local filename for `source`.

    HTTP(S) and SSH URLs are downloaded with :func:`fetch_url`; local
    paths are passed through unchanged (and never deleted).

    Parameters
    ----------
    source : str
        local path, HTTP(S) URL or SSH URL
    cache_file, cache_dir, force_download, session
        see :func:`fetch_url`
    autodelete : bool or None, optional
        if True, the downloaded copy (also in `cache_file`/`cache_dir`) is
        deleted when the context is left, i.e. after the file was read; a
        temporary directory is removed completely. If None (default), only
        a temporary directory (no `cache_file`/`cache_dir`) is removed.

    Yields
    ------
    str
        local filename
    """
    import shutil
    import tempfile

    if not is_url(source):  # also true for SSH URLs
        yield source
        return

    tmpdir = None
    if cache_file is None and cache_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="pypillometry_")
        cache_dir = tmpdir
    if autodelete is None:
        autodelete = tmpdir is not None
    fname = fetch_url(source, cache_file=cache_file, cache_dir=cache_dir,
                      force_download=force_download, session=session)
    try:
        yield fname
    finally:
        if autodelete:
            target = tmpdir or fname
            logger.info(f"Deleting downloaded copy: {target}")
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            elif os.path.exists(target):
                os.remove(target)


def write_pickle(obj, fname):
    """
    Store any Python object in a file using :mod:`pickle`.
    
    Parameters
    ----------
    obj : object
        object to save
    fname : str
        filename to save to
    """
    with open(fname, "wb") as f:
        pickle.dump(obj, f)
    

def read_pickle(
    fname: str,
    cache_file: Optional[str] = None,
    force_download: bool = False,
    session: Optional[requests.Session] = None,
    cache_dir: Optional[str] = None,
    autodelete: Optional[bool] = None):
    """
    Read a Python object from a file using :mod:`pickle`.

    URLs are downloaded with :func:`~pypillometry.io.fetch_url`, in the
    same way as in :func:`~pypillometry.io.read_eyelink`.
    
    Parameters
    ----------
    fname : str
        filename, HTTP(S) URL or SSH URL (``ssh://server/folder/file.pd``)
    cache_file : str, optional
        local filename to store a downloaded file in
    cache_dir : str, optional
        directory to store a downloaded file in (as
        ``cache_dir/<host>/<path>``), used if `cache_file` is None. If both
        are None, a temporary directory is used.
    force_download : bool, optional
        if True, download again even if a local copy exists in
        `cache_file`/`cache_dir`
    session : requests.Session, optional
        requests session for HTTP(S) downloads (e.g., from
        :func:`~pypillometry.io.osf_authenticate` for private projects)
    autodelete : bool or None, optional
        if True, the downloaded copy is deleted after it was read (also in
        `cache_file`/`cache_dir`). If None (default), only a temporary
        download (no `cache_file`/`cache_dir`) is deleted. Local files are
        never deleted.

    Returns
    -------
    object
        loaded object
    """
    with local_copy(fname, cache_file=cache_file, cache_dir=cache_dir,
                    autodelete=autodelete, force_download=force_download,
                    session=session) as local:
        with open(local, 'rb') as f:
            return pickle.load(f)

@requires_package("eyelinkio")
def read_eyelink(
    source: str, 
    cache_file: Optional[str] = None, 
    force_download: bool = False, 
    session: Optional[requests.Session] = None,
    cache_dir: Optional[str] = None,
    autodelete: Optional[bool] = None):
    """
    Read an Eyelink file/URL and return the object returned by the "eyelinkio" package.

    Requires the optional package `eyelinkio`. Its console output is only
    shown when the log level is "DEBUG". URLs are downloaded with
    :func:`~pypillometry.io.fetch_url`.
    
    Parameters
    ----------
    source : str
        filename of the Eyelink EDF file, HTTP(S) URL (OSF client URLs such
        as ``https://osf.io/<id>`` are converted to download URLs) or SSH URL
        (``ssh://server/folder/file.edf``)
    cache_file : str, optional
        local filename to store a downloaded file in
    cache_dir : str, optional
        directory to store a downloaded file in (as
        ``cache_dir/<host>/<path>``), used if `cache_file` is None. If both
        are None, a temporary directory is used.
    force_download : bool, optional
        if True, download again even if a local copy exists in
        `cache_file`/`cache_dir`
    session : requests.Session, optional
        requests session for HTTP(S) downloads (e.g., from
        :func:`~pypillometry.io.osf_authenticate` for private projects)
    autodelete : bool or None, optional
        if True, the downloaded copy is deleted after it was read (also in
        `cache_file`/`cache_dir`). If None (default), only a temporary
        download (no `cache_file`/`cache_dir`) is deleted. Local files are
        never deleted.

    Returns
    -------
    eyelinkio.EDF
        object returned by :func:`eyelinkio.read_edf`

    See Also
    --------
    pypillometry.GenericEyeData.from_eyelink : create an eyedata object from an EDF file
    """
    with local_copy(source, cache_file=cache_file, cache_dir=cache_dir,
                    autodelete=autodelete, force_download=force_download,
                    session=session) as fname:
        return _read_edf(fname)


def _read_edf(fname: str):
    """Read a local EDF file with eyelinkio (its output only shown at DEBUG level)."""
    import eyelinkio

    # Check if DEBUG logging is enabled
    current_level = logging_get_level()
    show_eyelinkio_output = (current_level == "DEBUG")

    if show_eyelinkio_output:
        logger.debug(f"Loading EDF file: {fname}")
        logger.debug("eyelinkio output will be displayed below:")
        edf = eyelinkio.read_edf(fname)
    else:
        logger.info(f"Loading EDF file: {fname} (current log level: {current_level}, set to DEBUG to see eyelinkio output)")
        with suppress_all_output():
            edf = eyelinkio.read_edf(fname)

    return edf