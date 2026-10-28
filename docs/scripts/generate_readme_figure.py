#!/usr/bin/env python
"""Generate the overview figure for the README (docs/images/readme_overview.png).

Both panels are drawn with pypillometry's own plotting functions. Left: a
stretch of raw pupil data, and the same data after blink interpolation and
low-pass filtering (`pupil_plot()`, blinks shaded). Right: the average pupil
response to the stimuli with its standard error (`SegmentedEyeData.plot()`).

Uses the "rlmw_002" example dataset (downloaded from OSF on the first call).

Usage:
    python docs/scripts/generate_readme_figure.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pypillometry as pp
from pypillometry import u

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "images", "readme_overview.png")

EYE = "left"
WINDOW = (124, 140)   # seconds shown in the left panel


def main():
    pp.logging_set_level("ERROR")
    d = pp.get_example_data("rlmw_002")
    raw = d.copy()
    d.pupil_blinks_detect(apply_mask=True)
    d.pupil_blinks_interpolate()
    d.pupil_lowpass_filter(cutoff=4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    # raw vs. cleaned pupil, detected blinks shaded
    plt.sca(ax1)
    raw.plot.pupil_plot(eyes=EYE, plot_range=WINDOW, units="sec",
                        show_events=False, show_blinks=False,
                        style={"color": "0.7", "linewidth": 1, "label": "raw"})
    d.plot.pupil_plot(eyes=EYE, plot_range=WINDOW, units="sec",
                      show_events=False, show_blinks=True,
                      style={"color": "C0", "linewidth": 2, "label": "cleaned"})
    ax1.set_title("Blinks detected, interpolated and filtered", loc="left")
    ax1.legend(loc="lower left", frameon=False)

    # event-related pupil dilation
    intervals = d.get_intervals("S_", (-500, 2500) * u.ms)
    erpd = d.get_segments(intervals, f"{EYE}_pupil").baseline_correct((-500, 0))
    erpd.plot(ax=ax2, show_legend=False)
    ax2.set_title(f"Pupil response to {erpd.n_segments} stimuli (mean ± SE)",
                  loc="left")

    fig.tight_layout(w_pad=3)
    fig.savefig(OUTPUT, dpi=150)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
