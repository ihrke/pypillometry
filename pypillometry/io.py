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
                if 'new_folder' in item['links']:
                    folder_url = item['links']['new_folder']
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
        If True, force re-download even if files exist locally. Default False.
    config_file : str, optional
        Name of the configuration file. Default is "pypillometry_conf.py"
    session : requests.Session, optional
        Authenticated session for accessing private projects. If None, uses unauthenticated requests.
        
    Returns
    -------
    study_data: dict
        Dictionary containing the loaded study data
    config: module
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

def load_study_local(path: str, config_file: str = "pypillometry_conf.py", subjects: list[str] = None):
    """
    Read a study from a local directory using the configuration file.
    
    Parameters
    ----------
    path : str
        Local path where the study data is stored
    config_file : str, optional
        Name of the configuration file. Default is "pypillometry_conf.py"
    subjects : list[str], optional
        List of subject IDs to load. If None, all subjects will be loaded.
        If a subject ID is provided that doesn't exist in the data, it will be skipped.
        
    Returns
    -------
    study_data: dict
        Dictionary containing the loaded study data
    config: module
        Module containing the configuration (pypillometry_conf.py imported as a module)
    """
    logger.info(f"Loading study from local directory '{path}'")
    
    # Check if path exists
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist")
        
    # First try to find config file in current directory
    if os.path.exists(config_file):
        logger.info(f"Found config file in current directory: {config_file}")
        config_path = config_file
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
    with change_dir(path):
        for subject_id in tqdm(subject_ids, desc="Loading subjects", unit="subject"):
            logger.debug(f"Loading subject {subject_id}")
            # Use the read_subject function from the config module
            info = config.raw_data[subject_id]
            # add local path to the paths in the info dict
            for key, value in info.items():
                if isinstance(value, str) and os.path.exists(value):
                    info[key] = value
            info["subject"] = subject_id
            study_data[subject_id] = config.read_subject(info)
        
    return study_data, config

def write_pickle(obj, fname):
    """
    Store any Python object in a file using :mod:`pickle`.
    
    Parameters
    ----------
    obj: object
        object to save
    fname: str
        filename to save to
    """
    with open(fname, "wb") as f:
        pickle.dump(obj, f)
    

def read_pickle(fname):
    """
    Read a Python object from a file using :mod:`pickle`.
    
    Parameters
    ----------
    fname: str
        filename or URL to load data from
        
    Returns
    -------
    object
        loaded object
    """
    if is_url(fname):
        # try loading from URL
        response = requests.get(fname, stream=True)
        if response.status_code == 200:
            total = int(response.headers.get('content-length', 0))
            content = bytearray()
            with tqdm(
                desc=f"Downloading {fname}",
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    content.extend(data)
                    bar.update(len(data))
            obj = pickle.loads(content)
    else:
        with open(fname, 'rb') as f:
            obj=pickle.load(f)
    return obj

@requires_package("eyelinkio")
def read_eyelink(
    source: str, 
    cache_file: str = None, 
    force_download: bool = False, 
    session: Optional[requests.Session] = None):
    """
    Read an Eyelink file/URL and return the object returned by the "eyelinkio" package.
    
    Parameters
    ----------
    source: str
        filename of the Eyelink file or URL
    cache_file: str, optional
        filename to cache the file in case it is downloaded from a URL
    force_download: bool, optional
        if True, force re-download even if file exists locally
    session: requests.Session, optional
        requests session to use for the download. 
    Returns
    -------
    object
        object returned by the "eyelinkio" package
    """
    import eyelinkio

    # Handle if source is a URL or local file
    if is_url(source):
        if cache_file is not None:
            # If cached file exists and no force_download, use it
            if os.path.exists(cache_file) and not force_download:
                fname = cache_file
            else:
                cache_dir = os.path.dirname(cache_file)
                # If cache_dir is non-empty and does not exist, make it
                if cache_dir and not os.path.exists(cache_dir):
                    logger.info(f"Creating directory for cache file: {cache_dir}")
                    os.makedirs(cache_dir, exist_ok=True)
                fname = download(source, cache_file, session=session)
        else:
            fname = download(source, session=session)
    else:
        fname = source

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