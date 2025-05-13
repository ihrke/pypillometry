.. currentmodule:: pypillometry


Sharing/Importing study data
============================

`pypillometry` provides functionality to easily load a complete study from a local cache directory and a user-provided configuration file. That way, scripts can avoid to include lengthy code for parsing the raw data files. Once the configuation script is in place, the user can load the data using the :func:`~pypillometry.load_study_local()` function:

.. code-block:: python

    import pypillometry as pp
    study,conf = pp.load_study_local(path="./data", config="pypillometry_conf.py")

In addition, you can upload the data and configuration file to the `Open Science Framework (OSF) <https://osf.io>`_ and use the following function to load the data:

.. code-block:: python

    study,conf = pp.load_study_osf(osf_id="your_project_id", path="./data")

(in that case, the data will be downloaded and cached in the specified path).

Creating a configuration file
------------------------------

A configuration file is a Python script that defines the data files, contains meta-data about the study and implements a function to read a dataset. The standard name for the configuration file is ``pypillometry_conf.py`` but you can use any name you want.

The configuration file (``pypillometry_conf.py``) should define:

- ``raw_data``: Dictionary mapping subject IDs to their data files
- ``read_subject()`` function that processes the raw data files

Here is an template for a configuration file (``pypillometry_conf.py``), see :ref:`examples/pypillometry_conf.py` for a complete example:

.. code-block:: python

    """
    Configuration file for my study.
    """
    import pypillometry as pp
    # other relevant imports

    # study metadata (optional)
    study_info = {
        "name": "My new study",
        "description": "...",
        ... # other metadata
    }

    # Dictionary of raw data files to be downloaded (required)
    # Keys are participant IDs, values are dictionaries containing paths to raw data files
    raw_data = {
        "001": {
            "events": "data/eyedata/asc/001_rlmw_events.asc",
            "samples": "data/eyedata/asc/001_rlmw_samples.asc"
        },   
        "002": {
            ...
        }
    }

    # Function to read a subject's data from the raw data files (required)
    def read_subject(info):
        """
        Read a subject's data from the raw data files.
        """
        # code for reading the data
        return pp.EyeData(...)

One you load the study data, the function `read_subject()` will be called each entry of the `raw_data` dictionary (usually one per subject but it can also be multiple entries per subject, e.g., if you have multiple tasks per subject). The way the `raw_data` dictionary and the `read_subject()` function are defined is completely up to you. Simply assure that the `read_subject()` function can process the data files specified in the `raw_data` dictionary.

The configuration file will be imported as a module and returned as the second argument of the `load_study_local()` and `load_study_osf()` functions. Therefore, you can access all variables and functions defined in the configuration file:

.. code-block:: python

    study,conf = pp.load_study_local(path="./data", config="pypillometry_conf.py")
    study["001"] # access data for subject 001
    conf.study_info # access study metadata
    conf.read_subject # access function to read data


The functions :func:`~pypillometry.io.load_study_local()` and :func:`~pypillometry.io.load_study_osf()` a;so allow to specify a list of subject codes that are to be loaded (default is to load all subjects).


Sharing Your Study on OSF
-------------------------

To share your study on OSF:

1. Create a new project on [Open Science Framework](https://osf.io)
2. Upload your study data files and configuration file (``pypillometry_conf.py``) to the project
    - the data files must be arranged in a folder structure that matches the one specified in the `raw_data` dictionary of the configuration file
3. Note down your project's OSF ID (found in the project URL)

To load a shared study from OSF, use :func:`~pypillometry.io.load_study_osf()`:

.. code-block:: python

    from pypillometry import load_study_osf
    
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
The function will:

1. Download the project's configuration file
2. Download the required data files for each subject
3. Process the data using the configuration's ``read_subject()`` function
4. Return a dictionary mapping subject IDs to their processed data

Files are cached locally in the specified path to avoid repeated downloads.

Example configuration file
--------------------------

Here is an example configuration file (``pypillometry_conf.py``) that could be used to share a study:

.. literalinclude:: ../examples/pypillometry_conf.py
   :language: python
   :linenos:

This configuration file is a real study that was shared on OSF (<https://osf.io/ca95r/>) and you can download the corresponding data using the following code (note that the size of the dataset is 3GB or so):

.. code-block:: python

    from pypillometry import load_study_osf
    study_data, config = load_study_osf("ca95r", path="./data")

The data will be downloaded and cached in the ``./data`` directory and `study_data` will be a dictionary mapping subject IDs to their data. The `config` object will be a module that contains the configuration file.