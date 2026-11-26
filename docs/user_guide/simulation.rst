.. currentmodule:: pypillometry.sim

Simulated data
==============

Simulated recordings are useful whenever the true signal has to be known:
to test a preprocessing method, to compare methods, or to check that a
model finds its own parameters again. :mod:`pypillometry.sim` builds such
recordings step by step and keeps the ground truth with the data.

.. admonition:: Experimental
   :class: experimental

   The latent pupil size is drawn from the timescale model
   (:doc:`/experimental/timescale_model`), which is experimental. The API
   of the simulations may still change.


A simulation is a recipe
------------------------

A :class:`Simulation` describes the recording (duration, sampling rate,
eyes, seed and, if needed, the experimental setup) and is built by chaining
steps. Each method returns a new simulation with one more step; the
recipe is only run by :meth:`~Simulation.run`:

.. code-block:: python

    import pypillometry as pp

    sim = (pp.sim.Simulation("5 min", fs=500, eyes=["left", "right"], seed=1)
           .pupil(binocular=True)       # latent pupil size (timescale model)
           .units(median=800)           # units of the eye tracker
           .noise(0.2))                 # white measurement noise (log-%)
    d = sim.run()                       # a SimEyeData object

Because each method returns a new simulation, a base recipe can be varied
without changing it:

.. code-block:: python

    base = pp.sim.Simulation("5 min", fs=500, seed=1).pupil().units(800)
    clean, noisy = base.noise(0.1).run(), base.noise(1.0).run()

The steps run in a fixed order, whatever the order in which they were
added. The signal is in log-% (:math:`100 \log(\text{size} /
\text{median})`) until the units step:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Stage
     - Method (step)
     - What it does
   * - latent pupil
     - ``.pupil()`` (:class:`TimescalePupil`)
     - fast component and slow components at fixed time scales, in log-%;
       with ``binocular=True`` common to both eyes plus a difference
       between them
   * - gaze
     - ``.gaze()`` (:class:`Fixations`)
     - fixations at random places on the screen
   * - foreshortening
     - ``.foreshortening()`` (:class:`Foreshortening`)
     - the pupil as seen by the camera, :math:`A_0 \cos\alpha`; needs the
       gaze and ``Simulation(setup=...)``
   * - units
     - ``.units(median)`` (:class:`Units`)
     - from log-% to the units of the tracker (default: median 1000)
   * - noise
     - ``.noise(sd)`` (:class:`WhiteNoise`)
     - white noise, multiplicative (``sd`` in log-%) or additive
       (``scale="data"``)

The parameters of :class:`TimescalePupil` are those of
:attr:`TimescaleModel.params <pypillometry.TimescaleModel.params>`;
parameters that are not given are the centre of the general prior,
i.e. typical values from real recordings.


The data and the ground truth
-----------------------------

:meth:`~Simulation.run` returns a :class:`SimEyeData` object. It is an
:class:`~pypillometry.EyeData` object, so all methods and plots work, with
two additions:

``d.truth``
    The true time series, one value per sample, in an
    :class:`~pypillometry.EyeDataDict`. They are cut, downsampled and
    saved together with the data, but the preprocessing does not change
    them:

    .. list-table::
       :widths: 25 75

       * - ``pupil``
         - the pupil as measured without noise (units of the data)
       * - ``size``
         - the true pupil size (differs from ``pupil`` with foreshortening)
       * - ``cosalpha``
         - the foreshortening factor
       * - ``log_size``
         - the latent signal in log-%, the sum of the components:
       * - ``log_fast``, ``log_diff``, ``log_3s``, ...
         - fast component, difference between the eyes, atoms (log-%)
       * - ``log_slow``
         - the atoms with time scales of at least the `cutoff` of
           :class:`TimescalePupil` (default 10 s)

``d.simulation``
    The recipe with the seed that was used. ``d.simulation.run()`` makes
    the same data again, ``d.simulation.run(seed=2)`` new data from the
    same recipe.

The plots ``d.plot.plot_truth()`` (data and true pupil) and
``d.plot.plot_components()`` (the true components) show the ground truth:

.. plot::
   :include-source: False

   sim = (pp.sim.Simulation("3 min", fs=50, seed=3).pupil().units(800).noise(0.5))
   d = sim.run()
   fig = d.plot.plot_components(figsize=(9, 6))


Random numbers
--------------

Each step draws its random numbers from its own stream, derived from the
seed of the simulation and the place of the step. Changing one step
therefore leaves the others unchanged: in the example above, ``clean``
and ``noisy`` have the same latent pupil size and differ only in the
noise. If a simulation has no seed, :meth:`~Simulation.run` draws one and
stores it in ``d.simulation``.


From a fitted model
-------------------

A fitted :class:`~pypillometry.TimescaleModel` simulates new recordings
with its estimated parameters, median pupil size and noise:

.. code-block:: python

    m = pp.TimescaleModel(d, eyes="left").fit()
    s = m.simulate("20 min", seed=1)           # SimEyeData at the rate of the model
    s = m.simulate("1 min", fs=1000, seed=1)   # at another sampling rate
    steps = m.process()                        # the latent pupil as steps

By default, the simulation has the sampling rate of the model (20 Hz).
At other rates, the standard deviation of the noise is scaled as for white
noise of the raw samples, which is only an approximation for real eye
trackers. Fitting the model again to simulated data shows how well its
parameters can be recovered from a recording of that length (see
:doc:`/notebooks/simulation`).


Saving
------

Simulated data are saved like any other data (``d.write_file()``,
:func:`~pypillometry.read_file`), together with the ground truth and the
recipe.
