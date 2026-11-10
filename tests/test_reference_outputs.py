"""
Reference outputs: what the current code computes.

Each case is one function that runs the API calls and returns a dict of
results (numpy arrays and plain scalars). The script
``tests/data/reference/make_reference.py`` ran the cases and stored the
results in ``tests/data/reference/`` (``<case id>.npz`` for the arrays,
``reference.json`` for the scalars and the description of the cases). The
tests here recompute every case and compare it with the stored results, so
that any change of a computed number is noticed.

When the API changes, only the calls inside the case functions (and the
helpers that read the results) are adapted. When a number changes on
purpose, the reference is regenerated with ``make_reference.py`` in the same
commit, after checking that the new values are right.

The data are pypillometry's packaged example data (``rlmw_002_short``) and
synthetic data generated here (fixed seeds).
"""
import functools
import json
import os
import warnings

import numpy as np
import pytest

import pypillometry as pp
from pypillometry import u

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reference")

#: tolerances for floating point values
RTOL = 1e-9
ATOL = 1e-9

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

#: NaN blocks (blinks) in the synthetic pupil data, in samples, end-exclusive
SYN_BLINKS = {"left": [(1000, 1060), (3000, 3150), (6000, 6040), (8000, 8006)],
              "right": [(1010, 1070), (5000, 5100), (8000, 8030)]}
#: off-screen periods in the synthetic gaze data: (eye, coordinate, start, end, value)
SYN_OFFSCREEN = [("left", "x", 2000, 2050, 1400.), ("left", "y", 4000, 4010, -20.),
                 ("right", "x", 7000, 7100, -50.)]
#: period in which the right eye diverges (x + 300 px), in samples
SYN_DIVERGENCE = (5500, 5700)


@functools.lru_cache(maxsize=None)
def _rlmw():
    with warnings.catch_warnings(), pp.nologging():
        warnings.simplefilter("ignore")
        return pp.get_rlmw_002_short()


@functools.lru_cache(maxsize=None)
def _synthetic():
    """Binocular EyeData, 20 s at 500 Hz, with NaN blinks and gaze artifacts."""
    n, fs = 10000, 500.
    t = np.arange(n) * 1000. / fs
    rng = np.random.default_rng(20261103)
    base = 1000 + 40 * np.sin(2 * np.pi * t / 4000.)
    data = {}
    for eye, fac in (("left", 1.0), ("right", 0.9)):
        p = fac * base + rng.normal(0, 0.5, n)
        for on, off in SYN_BLINKS[eye]:
            # the pupil closes and opens over 15 samples around the NaN block
            ramp = np.linspace(0, 1, 15)
            p[on - 15:on] -= 200 * ramp
            p[off:off + 15] -= 200 * ramp[::-1]
            p[on:off] = np.nan
        data[f"{eye}_pupil"] = p
        data[f"{eye}_x"] = 640 + 30 * np.sin(2 * np.pi * t / 3000.) + rng.normal(0, 2, n)
        data[f"{eye}_y"] = 512 + 20 * np.cos(2 * np.pi * t / 5000.) + rng.normal(0, 2, n)
    for eye, var, on, off, val in SYN_OFFSCREEN:
        data[f"{eye}_{var}"][on:off] = val
    data["right_x"][SYN_DIVERGENCE[0]:SYN_DIVERGENCE[1]] += 300
    onsets = np.arange(500., 19000., 1500.)
    labels = np.array([f"stim_{i}" for i in range(len(onsets))])
    onsets = np.concatenate([onsets, onsets + 800.])
    labels = np.concatenate([labels, [f"resp_{i}" for i in range(len(labels))]])
    order = np.argsort(onsets, kind="stable")
    setup = pp.ExperimentalSetup(screen_resolution=(1280, 1024),
                                 physical_screen_size=("30 cm", "20 cm"),
                                 eye_to_screen_center="60 cm")
    with pp.nologging():
        return pp.EyeData(time=t, sampling_rate=fs, event_onsets=onsets[order],
                          event_labels=labels[order], experimental_setup=setup,
                          name="synthetic", **data)


DATASETS = {"rlmw": _rlmw, "synth": _synthetic}
#: event used for the event-locked cases per data set
EVENT = {"rlmw": "S_", "synth": "stim"}


def get_data(dataset):
    """Fresh copy of a data set (`rlmw` or `synth`)."""
    return DATASETS[dataset]().copy()


def preprocessed(dataset):
    """Data with blinks detected (masked) and interpolated (case b)."""
    return get_data(dataset).pupil.blinks_detect().pupil.blinks_interpolate()


# ---------------------------------------------------------------------------
# helpers that read results from the objects
# ---------------------------------------------------------------------------

def iv_samples(iv):
    """Intervals (in samples) as int64 array (n, 2)."""
    return np.asarray(iv.intervals, dtype=np.int64).reshape(-1, 2)


def iv_float(iv):
    """Intervals (in time units) as float array (n, 2)."""
    return np.asarray(iv.intervals, dtype=float).reshape(-1, 2)


def pupil_keys(d):
    return [k for k in d.data.keys() if k.endswith("_pupil")]


def strict_mask(d, k):
    """The mask under the strict policy (interpolated samples are masked too)."""
    return d.mask(k, policy="strict")


def masks(d, keys=None):
    """Mask of every variable."""
    keys = d.data.keys() if keys is None else keys
    return {f"mask.{k}": strict_mask(d, k) for k in keys}


def blinks(d, keys, units="samples"):
    out = {}
    for k, iv in d.get_blinks(units=units).items():
        if k in keys:
            out[f"blinks.{k}"] = iv_samples(iv) if units == "samples" else iv_float(iv)
    return out


def interpolated_blinks(d, keys):
    return {f"interpolated.{k}": iv_samples(iv)
            for k, iv in d.get_interpolated(units="samples").items() if k in keys}


def summary(d, step=50):
    """Compact description of an object: time, events, data, masks, blinks."""
    out = {"n": int(len(d.tx)), "fs": float(d.fs), "tx_first": float(d.tx[0]),
           "tx_last": float(d.tx[-1]), "tx_sub": np.asarray(d.tx[::step], dtype=float),
           "event_onsets": np.asarray(d.events.onsets, dtype=float),
           "event_labels": np.asarray(d.events.labels, dtype=str)}
    for k in d.data.keys():
        v = np.asarray(d.data[k], dtype=float)
        m = strict_mask(d, k)
        out[f"nansum.{k}"] = float(np.nansum(v))
        out[f"masked.{k}"] = int(m.sum())
        out[f"sub.{k}"] = v[::step].copy()
        out[f"mask.{k}"] = m.copy()
    out.update(blinks(d, pupil_keys(d)))
    out.update(interpolated_blinks(d, pupil_keys(d)))
    return out


def masked_mean(a, axis=None):
    """Mean of a masked array, NaN where everything is masked."""
    return np.ma.filled(np.ma.mean(a, axis=axis).astype(float), np.nan)


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------

#: case id -> {"func", "description", "params"}
CASES = {}


def case(case_id, description, **params):
    """Register a function as a case (called with `params` as keyword arguments)."""
    def deco(f):
        assert case_id not in CASES, case_id
        CASES[case_id] = {"func": f, "description": description, "params": params}
        return f
    return deco


BLINK_PARAMS = {"default": {}, "zero": {"strategies": ["zero"]},
                "vel10": {"vel_onset": -10, "min_duration": 10}}


def _blinks_detect(dataset, params):
    d = get_data(dataset)
    d.pupil.blinks_detect(**params)
    out = {f"stored.{k}": iv_samples(iv) for k, iv in
           d.get_blinks(units="samples").items() if k in pupil_keys(d)}
    out.update({f"stored_ms.{k}": iv_float(iv) for k, iv in
                d.get_blinks(units="ms").items() if k in pupil_keys(d)})
    out.update(masks(d))
    if dataset == "synth":
        for eye, ivs in SYN_BLINKS.items():
            out[f"nan_blocks.{eye}_pupil"] = np.asarray(ivs, dtype=np.int64)
    return out


for _ds in DATASETS:
    for _name, _params in BLINK_PARAMS.items():
        case(f"a.blinks_detect.{_ds}.{_name}",
             f"pupil.blinks_detect({_params or ''}) on {_ds}: stored blink "
             "intervals (samples, ms) and masks",
             dataset=_ds, params=_params)(_blinks_detect)


def _blinks_interpolate(dataset):
    d = preprocessed(dataset)
    out = {f"pupil.{k}": np.asarray(d.data[k], dtype=float).copy() for k in pupil_keys(d)}
    out.update(masks(d))
    out.update(blinks(d, pupil_keys(d)))
    out.update(interpolated_blinks(d, pupil_keys(d)))
    return out


def _lowpass(dataset):
    d = preprocessed(dataset).pupil.lowpass_filter(5)
    out = {f"pupil.{k}": np.asarray(d.data[k], dtype=float).copy() for k in pupil_keys(d)}
    out.update(masks(d, pupil_keys(d)))
    return out


for _ds in DATASETS:
    case(f"b.blinks_interpolate.{_ds}",
         f"pupil.blinks_detect(), pupil.blinks_interpolate() on {_ds}: "
         "pupil values, masks, blinks, interpolated blinks (samples)",
         dataset=_ds)(_blinks_interpolate)
    case(f"c.lowpass.{_ds}",
         f"case b, then pupil.lowpass_filter(5) on {_ds}: pupil values and masks",
         dataset=_ds)(_lowpass)


def _offscreen(dataset):
    d = get_data(dataset)
    d.gaze.mask_offscreen()
    # the runs of the layer per eye (the same for x and y)
    found = d.get_layer_intervals("offscreen", variables="x", units="samples")
    out = {f"runs.{key.partition('_')[0]}": iv_samples(iv) for key, iv in found.items()}
    out.update(masks(d))
    return out


def _divergence(dataset, params):
    d = get_data(dataset)
    d.gaze.mask_divergences(**params)
    # the runs of the layer (the same for all gaze variables)
    out = {"runs": iv_samples(
        d.get_layer_intervals("divergence", "left", "x", units="samples")["left_x"])}
    out.update(masks(d))
    return out


DIVERGENCE_PARAMS = {"pct99": {}, "px100": {"threshold": 100, "thr_type": "pixel"}}

for _ds in DATASETS:
    case(f"d.offscreen.{_ds}",
         f"gaze.mask_offscreen() on {_ds} (screen 1280x1024): runs of the layer "
         "(samples) and masks",
         dataset=_ds)(_offscreen)
    for _name, _params in DIVERGENCE_PARAMS.items():
        case(f"d.divergence.{_ds}.{_name}",
             f"gaze.mask_divergences({_params or ''}) on {_ds}: runs of the layer "
             "(samples) and masks",
             dataset=_ds, params=_params)(_divergence)


def _segments(dataset, preproc):
    d = preprocessed(dataset) if preproc else get_data(dataset)
    ep = d.get_epochs(EVENT[dataset], window=(-200, 1000) * u.ms)
    out = {"intervals_ms": iv_float(ep.to_intervals()),
           "intervals_samples": iv_samples(ep.to_intervals("samples", d)),
           "event_onsets": np.asarray(ep.onsets, dtype=float)}
    for k in pupil_keys(d):
        seg = d.get_segments(ep, k)
        with pp.nologging():
            bc = seg.pupil.baseline_correct((-200, 0), inplace=False)
        out[f"tx.{k}"] = np.asarray(seg.tx, dtype=float)
        out[f"mask.{k}"] = np.ma.getmaskarray(seg[k]).copy()
        out[f"mean.{k}"] = masked_mean(seg[k], axis=0)
        out[f"bc_data.{k}"] = np.ma.getdata(bc[k]).astype(float).copy()
        out[f"bc_mask.{k}"] = np.ma.getmaskarray(bc[k]).copy()
        out[f"bc_mean.{k}"] = masked_mean(bc[k], axis=0)
    return out


def _stat_per_event(dataset, preproc):
    d = preprocessed(dataset) if preproc else get_data(dataset)
    out = {}
    for name, window in (("pre", (-200, 0)), ("post", (0, 1000))):
        st = d.stat_per_event(window=window * u.ms, event_select=EVENT[dataset])
        out.update({f"mean_{name}.{k}": np.asarray(v, dtype=float) for k, v in st.items()})
    st = d.stat_per_event(window=(0, 1000) * u.ms, event_select=EVENT[dataset],
                          variables="pupil", statfct=np.ma.median)
    out.update({f"median_post.{k}": np.asarray(v, dtype=float) for k, v in st.items()})
    return out


for _ds in DATASETS:
    for _pre in (False, True):
        _tag = "interp" if _pre else "raw"
        _what = "after case b" if _pre else "raw data"
        case(f"e.segments.{_ds}.{_tag}",
             f"get_epochs('{EVENT[_ds]}', window=(-200,1000)*u.ms), get_segments "
             f"per pupil, baseline_correct((-200,0)) on {_ds} ({_what}): intervals, "
             "segment tx, masks, data, per-segment means",
             dataset=_ds, preproc=_pre)(_segments)
        case(f"e.stat_per_event.{_ds}.{_tag}",
             f"stat_per_event(window=(-200,0)/(0,1000)*u.ms, event_select='{EVENT[_ds]}') "
             f"mean for all variables and np.ma.median for pupil on {_ds} ({_what})",
             dataset=_ds, preproc=_pre)(_stat_per_event)


def _quality(dataset, preproc):
    d = preprocessed(dataset) if preproc else get_data(dataset)
    q = d.quality_report().to_dict(flat=True)
    return {k: v for k, v in q.items()
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)}


for _ds in DATASETS:
    for _pre in (False, True):
        case(f"f.quality.{_ds}.{'interp' if _pre else 'raw'}",
             f"quality_report().to_dict(flat=True), numeric values, on {_ds} "
             f"({'after case b' if _pre else 'raw data'})",
             dataset=_ds, preproc=_pre)(_quality)


def _crop(dataset):
    return summary(preprocessed(dataset).crop(5000, 25000))


def _remove(dataset):
    d = preprocessed(dataset).remove_intervals([(10000, 12000), (30000, 31000)])
    out = summary(d)
    out["removed_intervals"] = np.asarray(d.params["removed_intervals"], dtype=float).reshape(-1, 2)
    return out


def _exclude(dataset):
    d = preprocessed(dataset).exclude_intervals([(0, 2000), (20000, 22500)])
    out = summary(d)
    out["excluded_ms"] = iv_float(d.get_excluded())
    out["excluded_samples"] = iv_samples(d.get_excluded(units="samples"))
    out["included_ms"] = iv_float(d.get_included())
    out["included_samples"] = iv_samples(d.get_included(units="samples"))
    return out


def _downsample(dataset, fsd):
    return summary(preprocessed(dataset).downsample(fsd), step=10)


case("g.crop.rlmw", "case b on rlmw, then crop(5000, 25000) (ms): summary "
     "(time, events, data every 50th sample, masks, blinks)", dataset="rlmw")(_crop)
case("g.remove_intervals.rlmw", "case b on rlmw, then remove_intervals([(10000,12000), "
     "(30000,31000)]) (ms): summary and params['removed_intervals']",
     dataset="rlmw")(_remove)
case("g.exclude_intervals.rlmw", "case b on rlmw, then exclude_intervals([(0,2000), "
     "(20000,22500)]) (ms): summary, get_excluded/get_included in ms and samples",
     dataset="rlmw")(_exclude)
for _fsd in (250, 100):
    case(f"g.downsample.rlmw.{_fsd}", f"case b on rlmw, then downsample({_fsd}) "
         "(integer factor): summary (data every 10th sample)",
         dataset="rlmw", fsd=_fsd)(_downsample)


def _scale(dataset):
    d = preprocessed(dataset).scale()
    out = {}
    for stat in ("center", "scale"):
        for key, v in d.params["scale"][stat].items():
            out[f"param_{stat}.{key}"] = float(v)
    for k in d.data.keys():
        x = np.ma.masked_array(d.data[k], mask=strict_mask(d, k))
        out[f"mean.{k}"] = float(masked_mean(x))
        out[f"std.{k}"] = float(np.ma.std(x))
        out[f"nanmean_all.{k}"] = float(np.nanmean(d.data[k]))
    return out


def _merge_eyes(dataset):
    d = preprocessed(dataset).merge_eyes()
    out = {}
    for k in d.data.keys():
        if not k.startswith("mean_"):
            continue
        x = np.ma.masked_array(d.data[k], mask=strict_mask(d, k))
        out[f"mean.{k}"] = float(masked_mean(x))
        out[f"nanmean_all.{k}"] = float(np.nanmean(d.data[k]))
        out[f"masked.{k}"] = int(strict_mask(d, k).sum())
        out[f"sub.{k}"] = np.asarray(d.data[k][::50], dtype=float)
    out["keys"] = np.asarray(sorted(d.data.keys()), dtype=str)
    return out


for _ds in DATASETS:
    case(f"h.scale.{_ds}", f"case b on {_ds}, then scale(): stored parameters, "
         "masked mean/std of every variable", dataset=_ds)(_scale)
    case(f"h.merge_eyes.{_ds}", f"case b on {_ds}, then merge_eyes(): merged "
         "variables (mean, masked samples, every 50th sample)", dataset=_ds)(_merge_eyes)


def compute(case_id):
    """Run a case and return its results (dict name -> array or scalar)."""
    c = CASES[case_id]
    with warnings.catch_warnings(), pp.nologging():
        warnings.simplefilter("ignore")
        return c["func"](**c["params"])


# ---------------------------------------------------------------------------
# storing and comparing
# ---------------------------------------------------------------------------

def split_results(res):
    """Split results into arrays (for the .npz) and scalars (for the JSON)."""
    arrays, scalars = {}, {}
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, (bool, np.bool_)):
            scalars[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            scalars[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            scalars[k] = float(v)
        else:
            raise TypeError(f"{k}: unsupported type {type(v)}")
    return arrays, scalars


def load_reference():
    with open(os.path.join(REF_DIR, "reference.json")) as f:
        return json.load(f)


def load_arrays(case_id):
    with np.load(os.path.join(REF_DIR, f"{case_id}.npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def assert_same(name, actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    if expected.dtype.kind in "biuUS":
        np.testing.assert_array_equal(actual, expected, err_msg=name)
    else:
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL,
                                   equal_nan=True, err_msg=name)


def test_reference_covers_all_cases():
    assert set(load_reference()["cases"]) == set(CASES)


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_reference_output(case_id):
    ref = load_reference()["cases"][case_id]
    arrays, scalars = split_results(compute(case_id))
    ref_arrays = load_arrays(case_id)
    assert set(arrays) == set(ref_arrays)
    assert set(scalars) == set(ref["scalars"])
    for k in sorted(ref_arrays):
        assert_same(f"{case_id}:{k}", arrays[k], ref_arrays[k])
    for k, v in sorted(ref["scalars"].items()):
        if isinstance(v, float):
            assert_same(f"{case_id}:{k}", scalars[k], v)
        else:
            assert scalars[k] == v, f"{case_id}:{k}: {scalars[k]} != {v}"
