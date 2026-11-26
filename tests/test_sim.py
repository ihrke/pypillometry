"""pp.sim: Simulation, its steps and SimEyeData."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pypillometry as pp
from pypillometry.sim import Simulation, TimescalePupil, WhiteNoise


def setup():
    return pp.ExperimentalSetup(screen_resolution=(1920, 1080),
                                physical_screen_size=("52 cm", "29 cm"),
                                eye_to_screen_center="60 cm",
                                camera_spherical=("20 deg", "-90 deg", "70 cm"))


def foreshortening_sim(duration="2 s", fs=100, seed=1):
    """the whole chain: pupil, gaze, foreshortening, units, noise"""
    return (Simulation(duration, fs=fs, seed=seed, setup=setup())
            .pupil().gaze().foreshortening().units(750).noise(0.5))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_builder_returns_new_simulations():
    base = Simulation("1 s", fs=100, seed=1).pupil()
    a, b = base.noise(0.1), base.noise(1.0)
    assert len(base.steps) == 1 and len(a.steps) == len(b.steps) == 2
    assert "WhiteNoise(sd=1)" in repr(b) and "seed=1" in repr(b)


def test_order_of_steps_does_not_matter():
    s = setup()
    a = Simulation("1 s", fs=100, seed=3, setup=s).pupil().gaze().foreshortening().units(500)
    b = Simulation("1 s", fs=100, seed=3, setup=s).units(500).foreshortening().gaze().pupil()
    np.testing.assert_array_equal(a.run()["left_pupil"], b.run()["left_pupil"])


def test_reproducible_and_seed_is_stored():
    d = Simulation("2 s", fs=100).pupil().noise().run()
    assert d.simulation.seed is not None
    np.testing.assert_array_equal(d.simulation.run()["left_pupil"], d["left_pupil"])
    assert not np.array_equal(d.simulation.run(seed=d.simulation.seed + 1)["left_pupil"],
                              d["left_pupil"])


def test_each_step_has_its_own_random_stream():
    base = Simulation("5 s", fs=100, seed=1).pupil().units(800)
    a, b = base.noise(0.1).run(), base.noise(2.0).run()
    np.testing.assert_array_equal(a.truth["left", "pupil"], b.truth["left", "pupil"])
    assert not np.array_equal(a["left_pupil"], b["left_pupil"])


def test_truth_of_the_timescale_pupil():
    d = Simulation("20 s", fs=50, seed=2).pupil(cutoff="30 s").units(800).noise(0.3).run()
    t = d.truth
    names = ["fast", "3s", "10s", "30s", "100s", "300s"]
    assert set(t.variables) == {"log_size", "log_slow", "size", "pupil"} | \
        {f"log_{n}" for n in names}
    np.testing.assert_allclose(t["left", "log_size"],
                               sum(t["left", f"log_{n}"] for n in names))
    np.testing.assert_allclose(t["left", "log_slow"], t["left", "log_30s"]
                               + t["left", "log_100s"] + t["left", "log_300s"])
    # no foreshortening: the measured size is the true size
    np.testing.assert_allclose(t["left", "size"], 800 * np.exp(t["left", "log_size"] / 100))
    np.testing.assert_array_equal(t["left", "size"], t["left", "pupil"])
    # multiplicative noise with sd 0.3 log-%
    e = 100 * np.log(d["left_pupil"] / t["left", "pupil"])
    assert abs(np.std(e) - 0.3) < 0.03


def test_parameters_of_the_timescale_pupil():
    d = Simulation("10 s", fs=50, seed=1).pupil(
        params={"fast_sd": 4.0, "sd_3s": 0.0, "sd_10s": 0.0, "sd_30s": 0.0,
                "sd_100s": 0.0, "sd_300s": 0.0}).run()
    assert np.abs(d.truth["left", "log_3s"]).max() < 1e-6
    assert 1 < np.std(d.truth["left", "log_fast"]) < 8
    with pytest.raises(ValueError, match="unknown parameters"):
        Simulation("1 s", fs=50).pupil(params={"nonsense": 1.0}).run()
    with pytest.raises(ValueError, match="more than one pupil step"):
        Simulation("1 s", fs=50).pupil().pupil().run()


def test_binocular():
    d = Simulation("10 s", fs=50, eyes=["left", "right"], seed=1).pupil(
        binocular=True, params={"diff_sd": 2.0, "gain": 1.0}).units(
        {"left": 1000, "right": 800}).run()
    t = d.truth
    np.testing.assert_allclose(t["left", "log_fast"], t["right", "log_fast"])
    np.testing.assert_allclose(t["left", "log_diff"], -t["right", "log_diff"])
    assert abs(np.ma.median(d["right_pupil"]) / np.ma.median(d["left_pupil"]) - 0.8) < 0.1
    # independent eyes
    i = Simulation("10 s", fs=50, eyes=["left", "right"], seed=1).pupil().run()
    assert not np.allclose(i.truth["left", "log_fast"], i.truth["right", "log_fast"])
    with pytest.raises(ValueError, match="needs two eyes"):
        Simulation("1 s", fs=50).pupil(binocular=True).run()


def test_foreshortening_chain():
    d = foreshortening_sim().run()
    t = d.truth
    assert d.experimental_setup.screen_resolution == (1920, 1080)
    np.testing.assert_allclose(t["left", "pupil"], t["left", "size"] * t["left", "cosalpha"])
    assert (t["left", "cosalpha"] < 1).all() and (t["left", "cosalpha"] > 0.5).all()
    x, y = d["left_x"], d["left_y"]
    assert (x >= 0).all() and (x <= 1920).all() and (y >= 0).all() and (y <= 1080).all()
    assert 3 <= len(np.unique(x)) <= 15         # fixations of about 300 ms
    with pytest.raises(ValueError, match="experimental setup"):
        Simulation("1 s", fs=50).pupil().gaze().foreshortening().run()
    with pytest.raises(ValueError, match="needs the gaze"):
        Simulation("1 s", fs=50, setup=setup()).pupil().foreshortening().run()


def test_checks():
    with pytest.raises(ValueError, match="eyes"):
        Simulation("1 s", eyes=["left", "middle"])
    with pytest.raises(ValueError, match="duration"):
        Simulation(0)
    with pytest.raises(ValueError, match="scale"):
        WhiteNoise(1.0, scale="dB")
    with pytest.raises(ValueError, match="no eyes"):
        Simulation("1 s", fs=50).pupil().add(WhiteNoise(eyes="right")).run()
    with pytest.raises(TypeError):
        Simulation("1 s").add("noise")


def test_steps_to_dict_and_back():
    for step in (TimescalePupil(params={"fast_sd": 3.0}, eyes=("left",), binocular=False),
                 WhiteNoise(0.4, "data", "left")):
        assert type(step).from_dict(step.to_dict()) == step


def test_truth_follows_the_samples():
    d = foreshortening_sim("4 s", fs=500).run()
    # the true pupil in the data, to check the alignment
    d.data["left_pupil"] = d.truth["left", "pupil"].copy()

    def check(x):
        for k in x.truth.keys():
            assert len(x.truth[k]) == len(x)
        np.testing.assert_array_equal(x.truth["left", "pupil"], x.data["left_pupil"])

    x = d.crop(500, 2500, inplace=False)
    assert len(x) == 1001
    check(x)
    check(d.remove_intervals([(1000, 1500)], inplace=False))
    check(d.downsample(50, inplace=False))
    assert len(d.truth["left", "pupil"]) == 2000      # original unchanged
    keep = np.ones(len(d), dtype=bool)
    keep[500:600] = False
    d._keep_samples(keep)                             # time gap of 200 ms
    d.fill_time_discontinuities()
    assert len(d) == 2000
    check(d)
    assert np.isnan(d.truth["left", "pupil"][550])


def test_class_and_repr():
    d = foreshortening_sim().run()
    assert isinstance(d, pp.EyeData) and type(d) is pp.sim.SimEyeData
    p = d.drop_data(variables=["x", "y"], inplace=False)
    assert type(p) is pp.sim.SimEyeData and p.variables == ["pupil"]
    r = repr(d)
    assert r.startswith("SimEyeData(") and " Simulation: Simulation(2 s at 100 Hz" in r
    assert " Truth: log_size" in r and r.index(" Truth:") < r.index(" History:")


def test_html_marks_simulated_data():
    d = foreshortening_sim().run()
    h = d._repr_html_()
    assert "SIMULATED" in h and "#f0a04b" in h
    assert "Simulation and ground truth" in h and "Foreshortening()" in h
    assert "cosalpha" in h
    real = pp.EyeData(left_pupil=np.ones(10), sampling_rate=10)._repr_html_()
    assert "SIMULATED" not in real and "1px solid #ddd" in real


def test_plots():
    d = Simulation("10 s", fs=50, eyes=["left", "right"], seed=1).pupil(
        binocular=True).units(800).noise().run()
    fig, ax = plt.subplots()
    assert d.plot.plot_truth(plot_range=(2, 5)) is ax and ax.has_data()
    nfigs = len(plt.get_fignums())
    fig = d.plot.plot_components(eye="right")
    assert isinstance(fig, plt.Figure) and len(plt.get_fignums()) == nfigs + 1
    fig2, axs = plt.subplots(1, 3)
    assert d.plot.plot_components(axes=axs) is fig2
    with pytest.raises(ValueError, match="axes needed"):
        d.plot.plot_components(axes=list(axs) + [axs[0]])
    d.plot.plot_pupil()                       # the plots of EyeData work
    g = foreshortening_sim().run()
    g.plot.plot_truth()
    with pytest.raises(ValueError, match="no latent pupil"):
        Simulation("1 s", fs=50).units().run().plot.plot_components()
