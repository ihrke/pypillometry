.. currentmodule:: pypillometry

Reading and writing data
========================

See :doc:`/user_guide/loading_data`. Eyelink files are usually read with
:meth:`EyeData.from_eyelink() <GenericEyeData.from_eyelink>`, saved
objects with :meth:`~GenericEyeData.from_file`.

Files and URLs
--------------

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   read_eyelink
   read_pickle
   write_pickle
   download
   fetch_url
   local_copy

Studies and OSF
---------------

See :doc:`/user_guide/studies`.

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   load_study_local
   load_study_osf
   get_osf_project_files
   osf_authenticate

Example datasets
----------------

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   get_available_datasets
   get_example_data
   get_rlmw_002
   get_rlmw_002_short
   get_rlmw_010_edf
