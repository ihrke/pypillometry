"""
io.py
=====

Read/Write data from/to disk.
"""

try:
   import cPickle as pickle
except:
   import pickle
import os
import requests
from contextlib import contextmanager
from tqdm import tqdm
from typing import Dict, Optional
from loguru import logger
from .convenience import is_url, suppress_all_output, requires_package
from .logging import logging_get_level

import requests

from pypillometry.convenience import change_dir


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
            total = int(resp.headers.get("content-length", 0))
            with open(fname, "wb") as file, tqdm(
                desc=os.path.basename(fname),
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


def _scp(url: str, target: str):
    """Copy the file (or folder, if `url` ends with ``/``) at an SSH URL to `target`."""
    import subprocess
    import shutil

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
    result = subprocess.run(cmd, capture_output=True, text=True)
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
    autodelete: bool = False,
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
    autodelete : bool, optional
        if True, the downloaded copy (also in `cache_file`/`cache_dir`) is
        deleted when the context is left, i.e. after the file was read; a
        temporary directory is removed completely. Default False.

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


def get_osf_project_files(osf_id: str, session: Optional[requests.Session] = None) -> Dict[str, Dict[str, str]]:
    """
    Get all file IDs from an OSF project.
    
    Parameters
    ----------
    osf_id : str
        The OSF project ID
    session : requests.Session, optional
        Authenticated session for accessing private projects. If None, uses unauthenticated requests.
        
    Returns
    -------
    Dict[str, Dict[str, str]]
        Dictionary mapping file paths to their IDs, download URLs, and file sizes
    """
    files = {}
    estimated_files = 100  # Start with estimate of 100 files
    
    def process_files(url: str, current_path: str = "", pbar=None) -> None:
        """
        Recursively process files and folders from an OSF API URL.
        
        Parameters
        ----------
        url : str
            The OSF API URL to process
        current_path : str
            Current path in the project structure
        pbar : tqdm
            Progress bar instance
        """
        nonlocal estimated_files
        
        if session is not None:
            response = session.get(url)
        else:
            response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Could not access project files: {response.status_code}")
            
        data = response.json()
        if 'data' not in data:
            raise ValueError(f"Unexpected API response structure: {data}")
            
        for item in data['data']:
            if 'attributes' not in item or 'links' not in item:
                continue
                
            attrs = item['attributes']
            if 'name' not in attrs or 'kind' not in attrs:
                continue
                
            name = attrs['name']
            full_path = os.path.join(current_path, name)
            
            if attrs['kind'] == 'file':
                if 'download' in item['links']:
                    # Get file size from attributes
                    size = attrs.get('size', 0)
                    files[full_path] = {
                        'id': item.get('id', ''),
                        'download_url': item['links']['download'],
                        'size': size
                    }
                    pbar.update(1)
                    # If we're close to the estimate, increase it
                    if len(files) >= estimated_files * 0.8:
                        estimated_files *= 2
                        pbar.total = estimated_files
                    pbar.set_description(f"Processing: {full_path}")
            elif attrs['kind'] == 'folder':
                # Use the 'upload' link and replace query params to list folder contents
                # The 'new_folder' link requires write permissions (403 for public access)
                if 'upload' in item['links']:
                    # upload link looks like: .../folder_id/?kind=file
                    # we need: .../folder_id/ (without query params)
                    folder_url = item['links']['upload'].split('?')[0]
                    process_files(folder_url, full_path, pbar)
                
        if isinstance(data.get('links'), dict) and data['links'].get('next'):
            process_files(data['links']['next'], current_path, pbar)
    
    # Process files with progress bar starting at estimated count
    root_url = f"https://api.osf.io/v2/nodes/{osf_id}/files/osfstorage/"
    with tqdm(total=estimated_files, desc="Processing files") as pbar:
        process_files(root_url, "", pbar)
    
    return files





def load_study_osf(osf_id: str, path: str, subjects: list[str] = None, force_download: bool = False, config_file: str = "pypillometry_conf.py", session: Optional[requests.Session] = None):
    """
    Read a study from OSF using the configuration file.
    Example: https://osf.io/p2u74/

    The configuration file (downloaded from the project) must define a
    dict `raw_data` (subject ID -> dict of data files) and a function
    `read_subject(info)` that returns the data for one subject. The
    listing of the project's files is cached in `path/info_osf.pkl` and
    data files are only downloaded if they are not yet in `path`.
    
    Parameters
    ----------
    osf_id : str
        The OSF project ID
    path : str
        Local path where files should be downloaded/stored
    subjects : list[str], optional
        List of subject IDs to load. If None, all subjects will be loaded.
        If a subject ID is provided that doesn't exist in the data, it will be skipped.
    force_download : bool, optional
        If True, force re-download (including the file listing) even if
        files exist locally. Default False.
    config_file : str, optional
        Name of the configuration file. Default is "pypillometry_conf.py"
    session : requests.Session, optional
        Authenticated session for accessing private projects. If None, uses unauthenticated requests.
        
    Returns
    -------
    study_data : dict
        Dictionary containing the loaded study data (subject ID -> result
        of `read_subject`)
    config : module
        Module containing the configuration (pypillometry_conf.py imported as a module)
    """
    logger.info(f"Loading study from OSF project '{osf_id}'")
    # Create cache directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    cache_file = os.path.join(path, "info_osf.pkl")

    # Try to load cached OSF file info
    if os.path.exists(cache_file) and not force_download:
        with open(cache_file, 'rb') as f:
            files = pickle.load(f)
        print("Using cached OSF file information")
    else:
        # Get all files in the project and estimate total download size
        print(f"Getting info on all files in project '{osf_id}'")
        files = get_osf_project_files(osf_id, session=session)
        
        # Cache the file information
        with open(cache_file, 'wb') as f:
            pickle.dump(files, f)
    
    # Calculate total download size from file sizes in API response
    total_size = sum(file_info['size'] for file_info in files.values())
    
    # Show download size/time if force_download
    if force_download:
        size_mb = round(total_size / (1024 * 1024), 1)  # Convert bytes to MB
        est_minutes = round(total_size / (1024 * 1024 * 60), 1)  # Convert bytes to minutes
        print(f"Total size: {size_mb} MB")
        print(f"Estimated download time (assuming 1MB/s): {est_minutes} minutes")
        
    # Find and download config file
    config_path = os.path.join(path, config_file)
    config_file_osf = next((f for f in files if f.endswith(config_file)), None)
    
    if config_file_osf is None:
        raise ValueError(f"Could not find {config_file} in project")
        
    if not os.path.exists(config_path) or force_download:
        if session is not None:
            response = session.get(files[config_file_osf]['download_url'])
        else:
            response = requests.get(files[config_file_osf]['download_url'])
        if response.status_code == 200:
            with open(config_path, 'wb') as f:
                f.write(response.content)
        else:
            raise ValueError(f"Could not download config file")
            
    # Load and parse config
    import importlib.util
    spec = importlib.util.spec_from_file_location("pypillometry_conf", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Filter subjects if specified
    subject_ids = list(config.raw_data.keys())
    if subjects is not None:
        subject_ids = [sid for sid in subjects if sid in config.raw_data]
        if not subject_ids:
            raise ValueError("None of the specified subjects were found in the data")
        print(f"Loading {len(subject_ids)} specified subjects")
    else:
        print(f"Loading all {len(subject_ids)} subjects")
    
    # Download and read raw data files
    for subject_id in subject_ids:
        subject_files = config.raw_data[subject_id]
        for file_type, data_file in subject_files.items():
            file_path = os.path.join(path, data_file)
            matching_file = next((f for f in files if f.endswith(data_file)), None)
            
            if matching_file is None:
                raise ValueError(f"Could not find {data_file} in project")
                
            if os.path.exists(file_path) and not force_download:
                continue
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if session is not None:
                response = session.get(files[matching_file]['download_url'], stream=True)
            else:
                response = requests.get(files[matching_file]['download_url'], stream=True)
            if response.status_code == 200:
                total = int(response.headers.get('content-length', 0))
                with open(file_path, 'wb') as f, tqdm(
                    desc=f"Downloading {data_file}",
                    total=total,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)
            else:
                raise ValueError(f"Could not download {data_file}")
                
    # apply the read_subject function from the config module
    study_data = {}

    for subject_id in subject_ids:
        # Use the read_subject function from the config module
        info = config.raw_data[subject_id]
        # add local path to the pathes in the info dict
        for key, value in info.items():
            if isinstance(value, str) and os.path.exists(os.path.join(path, value)):
                info[key] = os.path.join(path, value)
        info["subject"] = subject_id
        study_data[subject_id] = config.read_subject(info)
        
    return study_data, config

def load_study_local(path: str, config_file: str = "pypillometry_conf.py", subjects: list[str] = None,
                     cache_dir: Optional[str] = None, autodelete: bool = False, force_download: bool = False):
    """
    Read a study from a local directory (or an SSH URL) using the configuration file.

    The configuration file must define a dict `raw_data` (subject ID ->
    dict of data files, relative to `path`) and a function
    `read_subject(info)` that returns the data for one subject; it is
    called with the working directory changed to `path`.

    If `path` is an SSH URL (``ssh://server/folder/``, see
    :func:`fetch_url`), the configuration file and the data files of the
    selected subjects are downloaded to `cache_dir` before they are read.
    
    Parameters
    ----------
    path : str
        Local path or SSH URL where the study data is stored
    config_file : str, optional
        Name of the configuration file. Default is "pypillometry_conf.py".
        It is first looked up in the current working directory and, if
        not found there, in `path`.
    subjects : list[str], optional
        List of subject IDs to load. If None, all subjects will be loaded.
        If a subject ID is provided that doesn't exist in the data, it will be skipped.
    cache_dir : str, optional
        only for SSH URLs: directory to download the files to; if None, a
        temporary directory is used
    autodelete : bool, optional
        only for SSH URLs: if True, the data files of each subject are
        deleted after they were read (and a temporary directory is removed
        at the end). Default False.
    force_download : bool, optional
        only for SSH URLs: download files again even if they are already
        in `cache_dir`
        
    Returns
    -------
    study_data : dict
        Dictionary containing the loaded study data (subject ID -> result
        of `read_subject`)
    config : module
        Module containing the configuration (pypillometry_conf.py imported as a module)
    """
    import shutil
    import tempfile

    remote = None
    tmpdir = None
    if is_ssh_url(path):
        remote = path.rstrip("/") + "/"
        if cache_dir is None:
            tmpdir = tempfile.mkdtemp(prefix="pypillometry_")
            cache_dir = tmpdir
        local_path = _cache_path(remote, cache_dir)
        os.makedirs(local_path, exist_ok=True)
        logger.info(f"Loading study from '{path}' (files are downloaded to {local_path})")
    else:
        local_path = path
        logger.info(f"Loading study from local directory '{path}'")
        # Check if path exists
        if not os.path.exists(path):
            raise ValueError(f"Path '{path}' does not exist")

    try:
        # First try to find config file in current directory
        if os.path.exists(config_file):
            logger.info(f"Found config file in current directory: {config_file}")
            config_path = config_file
        elif remote is not None:
            config_path = fetch_url(remote + config_file, cache_dir=cache_dir,
                                    force_download=force_download)
        else:
            # If not found, look in the specified path
            config_path = os.path.join(path, config_file)
            if not os.path.exists(config_path):
                raise ValueError(f"Could not find {config_file} in current directory or in {path}")
            logger.info(f"Found config file in study directory: {config_path}")

        # Load and parse config
        logger.info("Loading configuration file")
        import importlib.util
        spec = importlib.util.spec_from_file_location("pypillometry_conf", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        # Filter subjects if specified
        subject_ids = list(config.raw_data.keys())
        if subjects is not None:
            subject_ids = [sid for sid in subjects if sid in config.raw_data]
            if not subject_ids:
                raise ValueError("None of the specified subjects were found in the data")
            logger.info(f"Loading {len(subject_ids)} specified subjects")
        else:
            logger.info(f"Loading all {len(subject_ids)} subjects")

        # Load the data using the read_subject function from the config module
        study_data = {}
        logger.info("Loading subject data")
        for subject_id in tqdm(subject_ids, desc="Loading subjects", unit="subject"):
            logger.debug(f"Loading subject {subject_id}")
            info = config.raw_data[subject_id]
            downloaded = []
            if remote is not None:
                for value in info.values():
                    if not isinstance(value, str):
                        continue
                    try:
                        downloaded.append(fetch_url(remote + value, cache_dir=cache_dir,
                                                    force_download=force_download))
                    except RuntimeError as e:
                        # entries in raw_data need not be files
                        logger.warning(f"Could not download '{value}' for subject {subject_id}: {e}")
            try:
                with change_dir(local_path):
                    # Use the read_subject function from the config module
                    info["subject"] = subject_id
                    study_data[subject_id] = config.read_subject(info)
            finally:
                if autodelete:
                    for fname in downloaded:
                        if os.path.exists(fname):
                            logger.debug(f"Deleting downloaded copy: {fname}")
                            os.remove(fname)
    finally:
        if autodelete and tmpdir is not None:
            logger.info(f"Deleting temporary directory: {tmpdir}")
            shutil.rmtree(tmpdir, ignore_errors=True)

    return study_data, config

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
    autodelete: bool = False):
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
    autodelete : bool, optional
        if True, the downloaded copy is deleted after it was read (also in
        `cache_file`/`cache_dir`). Local files are never deleted.
        Default False.

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
    autodelete: bool = False):
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
    autodelete : bool, optional
        if True, the downloaded copy is deleted after it was read (also in
        `cache_file`/`cache_dir`). Local files are never deleted.
        Default False.

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