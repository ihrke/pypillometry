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
