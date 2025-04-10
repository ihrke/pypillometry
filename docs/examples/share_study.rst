Sharing/Importing study data from OSF
====================================

`pypillometry` provides functionality to share and import study data via the `Open Science Framework (OSF) <https://osf.io>`_. The :func:`~pypillometry.load_study_osf()` function allows you to easily download and load study data that has been shared on OSF.

Sharing Your Study
-----------------

To share your study on OSF:

1. Create a new project on OSF
2. Upload your study data files and configuration file (``pypillometry_conf.py``) to the project
    - see 
3. Note down your project's OSF ID (found in the project URL)

The configuration file (``pypillometry_conf.py``) should define:

- ``raw_data``: Dictionary mapping subject IDs to their data files
- A ``read_subject()`` function that processes the raw data files

Loading Shared Data
------------------

To load a shared study, use :func:`~pypillometry.io.load_study_osf()`:

.. code-block:: python

    from pypillometry.io import load_study_osf
    
    # Load all subjects
    study_data = load_study_osf(
        osf_id="your_project_id",
        path="local/cache/path"
    )
    
    # Load specific subjects
    study_data = load_study_osf(
        osf_id="your_project_id", 
        path="local/cache/path",
        subjects=["sub01", "sub02"]
    )

Parameters
----------

- ``osf_id`` (str): The OSF project ID
- ``path`` (str): Local path where files should be downloaded/stored
- ``subjects`` (list[str], optional): List of specific subject IDs to load
- ``force_download`` (bool, optional): Force re-download of files even if they exist locally

The function will:

1. Download the project's configuration file
2. Download the required data files for each subject
3. Process the data using the configuration's ``read_subject()`` function
4. Return a dictionary mapping subject IDs to their processed data

Files are cached locally in the specified path to avoid repeated downloads.

Example configuration file
-------------------------
Here is an example configuration file (``pypillometry_conf.py``) that could be used to share a study:

.. literalinclude:: ../examples/pypillometry_conf.py
   :language: python
   :linenos:

This configuration file is a real study that was shared on OSF and you can download the corresponding data using the following code:

.. code-block:: python

    from pypillometry import load_study_osf
    study_data = load_study_osf("ca95r", path="./data")

The data will be downloaded and cached in the ``./data`` directory and `study_data` will be a dictionary mapping subject IDs to their data.