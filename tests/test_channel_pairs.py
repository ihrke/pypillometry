"""Selection of existing (eye, variable) pairs and data with incomplete pairs."""
import numpy as np
import pytest

from pypillometry import EyeDataDict


@pytest.fixture
def incomplete_dict():
    """Left eye with pupil and gaze, right eye with pupil only."""
    n = 5
    return EyeDataDict(left_pupil=np.ones(n), left_x=np.ones(n),
                       left_y=np.ones(n), right_pupil=np.ones(n))


class TestSelect:
    def test_all_pairs(self, incomplete_dict):
        assert incomplete_dict.select() == [
            ("left", "pupil"), ("left", "x"), ("left", "y"), ("right", "pupil")]
        assert incomplete_dict.select([], []) == incomplete_dict.select()

    def test_only_existing_pairs(self, incomplete_dict):
        assert incomplete_dict.select(variables=["x", "y"]) == [("left", "x"), ("left", "y")]
        assert incomplete_dict.select(eyes="right") == [("right", "pupil")]
        assert incomplete_dict.select(variables="pupil") == [("left", "pupil"), ("right", "pupil")]

    def test_order_follows_request(self, incomplete_dict):
        assert incomplete_dict.select(["right", "left"], "pupil") == [
            ("right", "pupil"), ("left", "pupil")]

    def test_missing_pair_raises(self, incomplete_dict):
        with pytest.raises(KeyError) as err:
            incomplete_dict.select(["left", "right"], ["pupil", "x"])
        msg = str(err.value)
        assert "right_x" in msg
        assert "left_pupil, left_x, left_y, right_pupil" in msg

    def test_missing_eye_raises(self, incomplete_dict):
        with pytest.raises(KeyError, match="mean"):
            incomplete_dict.select(eyes=["left", "mean"])
        # eye exists, but not with the requested variables
        with pytest.raises(KeyError, match="right"):
            incomplete_dict.select(eyes="right", variables=["x"])

    def test_missing_variable_raises(self, incomplete_dict):
        with pytest.raises(KeyError, match="baseline"):
            incomplete_dict.select(variables=["pupil", "baseline"])

    def test_empty_dict(self):
        assert EyeDataDict().select() == []


class TestEyeNameValidation:
    def test_tuple_key_with_underscore_in_eye(self):
        d = EyeDataDict()
        with pytest.raises(ValueError, match="may not contain '_'"):
            d["my_eye", "pupil"] = np.ones(3)
        with pytest.raises(ValueError, match="may not contain '_'"):
            d.set_with_mask(("my_eye", "pupil"), np.ones(3))
        assert len(d.data) == 0

    def test_string_key_splits_at_first_underscore(self):
        # a string key is never ambiguous: the eye is everything before the first '_'
        d = EyeDataDict(left_pupil_interp=np.ones(3))
        assert d.eyes == ["left"]
        assert d.variables == ["pupil_interp"]
        assert d.select("left", "pupil_interp") == [("left", "pupil_interp")]

    def test_tuple_read_with_underscore_in_eye(self):
        # "my_eye_pupil" is eye "my", variable "eye_pupil"; ("my_eye", "pupil")
        # must not silently return it
        d = EyeDataDict(my_eye_pupil=np.ones(3))
        assert d["my", "eye_pupil"] is d.data["my_eye_pupil"]
        with pytest.raises(ValueError, match="'my_eye'"):
            d["my_eye", "pupil"]

    def test_select_with_underscore_in_eye(self):
        d = EyeDataDict(my_eye_pupil=np.ones(3))
        with pytest.raises(ValueError, match="'my_eye'"):
            d.select("my_eye", "pupil")

    def test_eyedata_tuple_keys(self):
        import pypillometry as pp
        d = pp.EyeData(left_x=np.ones(3), left_y=np.ones(3), left_pupil=np.ones(3),
                       sampling_rate=10)
        with pytest.raises(ValueError, match="may not contain '_'"):
            d["my_eye", "pupil"] = np.ma.array(np.ones(3))
        with pytest.raises(ValueError, match="may not contain '_'"):
            d["my_eye", "pupil"]
        d["mean", "pupil"] = np.ones(3)
        assert "mean_pupil" in d.data
