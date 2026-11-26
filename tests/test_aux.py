"""Additional channels (`aux`) of EyeData objects."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pypillometry as pp


@pytest.fixture
def data():
    """10 s at 100 Hz, a gap of 200 ms (samples 500-519) and one aux channel
    that holds the time in ms (so that its alignment can be checked)."""
    tx = np.arange(1000) * 10.
    keep = np.r_[0:500, 520:1000]
    pupil = 5. + np.sin(tx[keep] / 500.)
    return pp.EyeData(time=tx[keep], left_pupil=pupil, sampling_rate=100,
                      aux={"env_temp": tx[keep]}, name="d")


class TestAux:
    def test_constructor(self, data):
        assert list(data.aux.keys()) == ["env_temp"]
        # the gap is filled in the aux channels too
        assert len(data) == 1000 and len(data.aux["env_temp"]) == 1000
        assert np.all(np.isnan(data.aux["env_temp"][500:520]))
        np.testing.assert_array_equal(data.aux["env_temp"][520:], data.tx[520:])

    def test_wrong_channels(self):
        with pytest.raises(ValueError, match="one value per sample"):
            pp.EyeData(left_pupil=np.ones(10), sampling_rate=10, aux={"a_b": np.ones(9)})
        with pytest.raises(ValueError, match="<group>_<name>"):
            pp.EyeData(left_pupil=np.ones(10), sampling_rate=10, aux={"temp": np.ones(10)})

    def test_empty_by_default(self):
        d = pp.EyeData(left_pupil=np.ones(10), sampling_rate=10)
        assert len(d.aux.keys()) == 0 and "aux" not in d.summary()
        assert len(d.crop(0, 500)) == 6

    def test_follows_samples(self, data):
        d = data.crop(2000, 3000, inplace=False)
        np.testing.assert_array_equal(d.aux["env_temp"], d.tx)
        d = data.remove_intervals(pp.Intervals([(1000, 2000)], units="ms"), inplace=False)
        assert len(d.aux["env_temp"]) == len(d)
        d = data.downsample(50, inplace=False)
        assert len(d.aux["env_temp"]) == len(d)
        assert d.aux["env_temp"][1] == pytest.approx(25.)  # mean of 20 and 30 ms

    def test_not_processed(self, data):
        """Methods for the data of the eyes leave the aux channels alone."""
        d = data.scale(inplace=False)
        np.testing.assert_array_equal(d.aux["env_temp"], data.aux["env_temp"])
        d = data.pupil.lowpass_filter(cutoff=4, inplace=False)
        np.testing.assert_array_equal(d.aux["env_temp"], data.aux["env_temp"])

    def test_copy(self, data):
        d = data.copy()
        d.aux["env_temp"][0] = -1
        assert data.aux["env_temp"][0] == 0

    @pytest.mark.parametrize("ext", ["pkl", "h5"])
    def test_save_and_load(self, data, ext, tmp_path):
        data.write_file(str(tmp_path / f"d.{ext}"))
        d = pp.EyeData.from_file(str(tmp_path / f"d.{ext}"))
        np.testing.assert_array_equal(d.aux["env_temp"], data.aux["env_temp"])

    def test_display(self, data):
        assert data.summary()["aux"] == ["env_temp"]
        assert "env_temp" in repr(data)
        assert "Aux channels" in data._repr_html_()
        empty = pp.EyeData(left_pupil=np.full(1000, 5.), sampling_rate=100)
        assert int(data.get_size()) > int(empty.get_size())

    def test_plots(self, data):
        fig = data.plot.plot_timeseries(variables=["pupil", "env_temp"])
        assert len(fig.axes) == 2
        ax = data.plot.plot_pupil_covariate("env_temp")
        assert ax.get_xlabel() == "env_temp"
        # all channels of a group in one panel
        assert len(data.plot.plot_timeseries(variables=["env"]).axes) == 1
        with pytest.raises(ValueError, match="no channel"):
            data.plot.plot_pupil_covariate("other")
        plt.close("all")

    def test_simulated_data(self):
        sim = pp.sim.Simulation(duration="10 s", fs=50, seed=1)
        d = sim.run()
        d._set_aux("env_temp", d.tx.copy())
        d.crop(2000, 3000)
        np.testing.assert_array_equal(d.aux["env_temp"], d.tx)
        assert len(d.truth[list(d.truth.keys())[0]]) == len(d)
