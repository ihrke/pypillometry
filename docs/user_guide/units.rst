.. currentmodule:: pypillometry

Units
=====

Times, distances and angles can be given with units. Multiply a number,
tuple or array by a unit from :data:`pypillometry.u`, or write it as a
string:

.. code-block:: python

    import pypillometry as pp
    from pypillometry import u

    d.sub_slice(3 * u.min, 5 * u.min)               # or ("3 min", "5 min")
    d.get_intervals("stim", (-200, 800) * u.ms)
    d.pupil_blinks_detect(min_duration="20 ms")
    d.plot.pupil_plot(plot_range=(3, 5) * u.min)
    d.set_experimental_setup(eye_to_screen_center=60 * u.cm,
                             screen_pitch="-5 deg")

Available units:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Quantity
     - Units
   * - time
     - ``u.ms``, ``u.sec``, ``u.min``, ``u.h``, ``u.samples`` (converted
       with the sampling rate)
   * - distance
     - ``u.mm``, ``u.cm``, ``u.m``, ``u.inch``
   * - angle
     - ``u.deg``, ``u.rad``

Aliases such as ``u.seconds``, ``u.s`` or ``u.degree`` work as well.

Internally, everything is stored in ms, mm and radians. Plain numbers
(without units) are interpreted as described in each function's
documentation, usually ms for times. Many functions also take a ``units=``
argument that says how plain numbers should be read, or in which units a
result should be returned:

.. code-block:: python

    d.get_duration(units="min")
    d.get_events(units="sec")
    d.sub_slice(3, 5, units="min")

To convert values yourself, use :func:`parse_time`,
:func:`parse_distance` and :func:`parse_angle`, or the
:meth:`Quantity.to` method:

.. code-block:: python

    pp.parse_time("2 min")              # 120000.0 (ms)
    (2 * u.min).to("sec")               # 120.0 sec
    pp.parse_angle("90 deg")            # 1.5707963...
