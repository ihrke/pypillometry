"""
study.py
========

Load a complete study (all subjects) from a local directory, an SSH URL or
an OSF project, based on a user-provided configuration file
(``pypillometry_conf.py``) that defines a dict `raw_data` and a function
`read_subject(info)`.
"""

import os
import pickle
import requests
from tqdm import tqdm
from typing import Dict, Optional
from loguru import logger

from .convenience import change_dir
from .io import fetch_url, is_ssh_url, _cache_path

__all__ = [
    "get_osf_project_files", "load_study_osf", "load_study_local",
]


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
        logger.info("Using cached OSF file information")
    else:
        # Get all files in the project and estimate total download size
        logger.info(f"Getting info on all files in project '{osf_id}'")
        files = get_osf_project_files(osf_id, session=session)
        
        # Cache the file information
        with open(cache_file, 'wb') as f:
            pickle.dump(files, f)
    
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
        logger.info(f"Loading {len(subject_ids)} specified subjects")
    else:
        logger.info(f"Loading all {len(subject_ids)} subjects")
    
    # Find raw data files that need to be downloaded
    to_download = []
    for subject_id in subject_ids:
        subject_files = config.raw_data[subject_id]
        for file_type, data_file in subject_files.items():
            file_path = os.path.join(path, data_file)
            matching_file = next((f for f in files if f.endswith(data_file)), None)
            
            if matching_file is None:
                raise ValueError(f"Could not find {data_file} in project")
                
            if os.path.exists(file_path) and not force_download:
                continue
            to_download.append((data_file, file_path, matching_file))

    # Show download size/time (from file sizes in API response)
    if to_download:
        total_size = sum(files[m]['size'] for _, _, m in to_download)
        size_mb = round(total_size / (1024 * 1024), 1)  # Convert bytes to MB
        est_minutes = round(total_size / (1024 * 1024 * 60), 1)  # at 1 MB/s
        logger.info(f"Downloading {len(to_download)} files, total size: {size_mb} MB "
                    f"(estimated download time at 1 MB/s: {est_minutes} minutes)")

    # Download raw data files
    for data_file, file_path, matching_file in to_download:
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
    :func:`~pypillometry.io.fetch_url`), the configuration file and the data files of the
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
