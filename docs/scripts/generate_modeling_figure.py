"""
Make docs/images/modeling_baseline_response.png (user guide, "Modeling").

The baseline is estimated with a Stan model, which needs CmdStan, so this
figure is not built with the documentation. Run it from the root of the
repository after a change of the modeling methods::

    python docs/scripts/generate_modeling_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pypillometry as pp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images",
                   "modeling_baseline_response.png")

pp.logging_set_level("WARNING")
d = (pp.get_example_data("rlmw_002_short") | pp.pupil.blinks_detect()
     | pp.pupil.blinks_interpolate() | pp.pupil.lowpass_filter(5))
d.pupil.estimate_baseline(eyes="left")
d.pupil.estimate_response(eyes="left", verbose=0)

t = d.tx / 1000
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(t, d.data["left_pupil"], color="0.6", lw=1, label="pupil")
ax.plot(t, d.data["left_baseline"], color="C0", lw=1.5, label="baseline (tonic)")
ax.plot(t, d.data["left_baseline"] + d.data["left_response"], color="C1", lw=1.2,
        label="baseline + responses to the events (model)")
ax.set(xlabel="time (s)", ylabel="left pupil")
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.tight_layout()
fig.savefig(OUT, dpi=100)
print("written", os.path.abspath(OUT))
