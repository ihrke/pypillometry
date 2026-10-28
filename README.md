<p align="center">
  <img src="https://raw.githubusercontent.com/ihrke/pypillometry/master/logo/pypillometry_logo_200x200.png" alt="pypillometry logo" width="160">
</p>

<h1 align="center">pypillometry</h1>

<p align="center">
  <b>Preprocessing, analysis and visualization of pupillometric and eye-tracking data in Python.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/pypillometry/"><img src="https://img.shields.io/pypi/v/pypillometry.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/pypillometry/"><img src="https://img.shields.io/pypi/pyversions/pypillometry.svg" alt="Python versions"></a>
  <a href="https://github.com/ihrke/pypillometry/actions/workflows/run-tests.yml"><img src="https://github.com/ihrke/pypillometry/actions/workflows/run-tests.yml/badge.svg" alt="tests"></a>
  <a href="https://codecov.io/gh/ihrke/pypillometry"><img src="https://codecov.io/gh/ihrke/pypillometry/graph/badge.svg?token=BFSKNGT772" alt="coverage"></a>
  <a href="https://ihrke.github.io/pypillometry/"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="documentation"></a>
  <br>
  <a href="https://doi.org/10.21105/joss.02348"><img src="https://joss.theoj.org/papers/3b06f4f3d5b703fd99c7e622b7edebe4/status.svg" alt="JOSS paper"></a>
  <a href="https://doi.org/10.5281/zenodo.3925528"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3925528.svg" alt="DOI"></a>
  <a href="https://github.com/ihrke/pypillometry/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ihrke/pypillometry.svg" alt="license"></a>
</p>

<p align="center">
  <a href="https://ihrke.github.io/pypillometry/">Documentation</a> ·
  <a href="https://ihrke.github.io/pypillometry/getting_started/quickstart.html">Quickstart</a> ·
  <a href="https://ihrke.github.io/pypillometry/tutorials/index.html">Tutorials</a> ·
  <a href="https://ihrke.github.io/pypillometry/api/index.html">API reference</a>
</p>

---

**pypillometry** reads recordings from eye trackers (e.g., Eyelink EDF files),
cleans them (blinks, artifacts, filtering), cuts them into event-related
segments and plots the results. Pupil size and gaze position of one or both
eyes are kept together in a single object, and every processing step is
recorded in the object's history.

![Raw and cleaned pupil data, and the average pupil response to 300 stimuli](https://raw.githubusercontent.com/ihrke/pypillometry/master/docs/images/readme_overview.png)

## Installation

```bash
pip install pypillometry
```

pypillometry requires Python 3.11 or newer. See the
[installation guide](https://ihrke.github.io/pypillometry/getting_started/installation.html)
for the development version and conda environments.

## Quick example

From raw data to an event-related pupil response, using one of the bundled
example datasets:

```python
import pypillometry as pp
from pypillometry import u

d = pp.get_example_data("rlmw_002")     # or: pp.EyeData.from_eyelink("recording.edf")
d.pupil_blinks_detect(apply_mask=True)  # detect and mask blinks
d.pupil_blinks_interpolate()            # interpolate them
d.pupil_lowpass_filter(cutoff=4)        # 4 Hz low-pass filter

# pupil response from 500 ms before to 2.5 s after each stimulus
intervals = d.get_intervals("S_", (-500, 2500) * u.ms)
erpd = d.get_segments(intervals, "left_pupil").baseline_correct((-500, 0))
erpd.plot()
```

The [quickstart](https://ihrke.github.io/pypillometry/getting_started/quickstart.html)
walks through this example step by step.

## Features

- **Reading data** from Eyelink EDF and ASC files, saved pickle files, and
  whole studies from a local folder or [OSF](https://osf.io)
- **Binocular data**: pupil size and gaze of both eyes in one object
- **Blinks**: detection, merging and interpolation (Mathôt, 2013)
- **Filtering**: low-pass filtering, smoothing, downsampling, scaling
- **Events and intervals** with flexible selection by label, function or time
- **Event-related pupil dilation** for single subjects and groups
- **Units** everywhere: `3 * u.min`, `"200 ms"`, `60 * u.cm`
- **Interactive viewer** for long recordings, and command-line tools
- **Modeling** of tonic baseline and phasic pupil responses (Mittner & Groot, 2025)

## Interactive viewer and command-line tools

`pp.view(d)` opens a fast, zoomable viewer for long recordings; `pp.tweak()`
lets you adjust preprocessing parameters while watching the result. The same
viewer is available from the command line:

```bash
pypilview recording.edf            # view an Eyelink file or a saved object
edf2pypil recording.edf            # convert EDF once, load much faster later
pypilinfo data/*.pkl               # print a summary of one or more files
```

![The pypilview viewer showing pupil size and gaze of both eyes](https://raw.githubusercontent.com/ihrke/pypillometry/master/docs/images/viewer_eyedata.png)

See [interactive tools](https://ihrke.github.io/pypillometry/user_guide/viewer.html)
and [command-line programs](https://ihrke.github.io/pypillometry/getting_started/cli.html)
in the documentation.

## Coming from version 1?

Version 2 is a major restructuring: it handles eye-tracking data in addition
to pupillometry, and more than one eye at a time. Scripts written for
version 1 need to be updated (see the
[changelog](https://ihrke.github.io/pypillometry/development/changelog.html)).
Version 1 is still available on the
[v1 branch](https://github.com/ihrke/pypillometry/tree/v1), with its
[documentation](https://ihrke.github.io/pypillometry/v1/html/index.html), and
can be installed with `pip install "pypillometry<2"`.

## Citing pypillometry

If you use pypillometry in your research, please cite

> Mittner, M. (2020). pypillometry: A Python package for pupillometric
> analyses. *Journal of Open Source Software*, 5(51), 2348.
> <https://doi.org/10.21105/joss.02348>

If you use the estimation of tonic and phasic pupil components, please also
cite

> Mittner, M., & Groot, J. M. (2025). A novel method for modeling tonic and
> phasic pupil dynamics in humans. *Behavior Research Methods*, 57(9), 233.
> <https://doi.org/10.3758/s13428-025-02755-7>

## Contributing

Bug reports and feature requests are welcome on the
[issue tracker](https://github.com/ihrke/pypillometry/issues). See the
[development guide](https://ihrke.github.io/pypillometry/development/index.html)
for how to contribute code.

pypillometry is maintained by Matthias Mittner (<matthias.mittner@uit.no>)
and released under the [MIT license](https://github.com/ihrke/pypillometry/blob/master/LICENSE).
