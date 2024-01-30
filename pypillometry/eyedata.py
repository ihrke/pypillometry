"""
eyedata.py
============

Implement EyeData class for use with the pypillometry package.
This class allows to store eyetracking and pupil data in a single object.
"""
from .eyedata_generic import GenericEyedata


class EyeData(GenericEyedata):
    def __init__(self, x=1):
        self.x = x

    def summary(self):
        print("This is a summary of the data 2.")