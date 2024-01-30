"""
eyedata.py
============

Implement EyeData class for use with the pypillometry package.
This class allows to store eyetracking and pupil data in a single object.
"""
from .eyedata_generic import GenericEyedata


class EyeData(GenericEyedata):
    def __init__(self, sampling_rate: float):
        self.fs = sampling_rate

    def summary(self):
        return {"blb": "blb"}