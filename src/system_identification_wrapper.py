import numpy as np
import matplotlib.pyplot as plt

try:
    from sippy_unipi import system_identification
except ImportError:
    import os
    import sys

    sys.path.append(os.pardir)
    from sippy_unipi import system_identification

from sippy_unipi import functionset as fset
from sippy_unipi import functionsetSIM as fsetSIM

class SystemIdentificationWrapper:
    def __init__(self, timeInput, timeOutput):
        self.outputValues            = []
        self.inputValues             = []
        self.interpolatedInputValues = []

        self.timeInput  = timeInput
        self.timeOutput = timeOutput

    def addInputData(self, inputData):
        if len(self.inputValues) == 0:
            self.inputValues = np.array([inputData])
        else:
            self.inputValues = np.vstack((self.inputValues, inputData))

    def addOutputData(self, outputData):
        if len(self.outputValues) == 0:
            self.outputValues = np.array([outputData])
        else:
            self.outputValues = np.vstack((self.outputValues, outputData))


    def buildInterpolatedInputValues(self):
        for data in self.inputValues:
            if len(self.interpolatedInputValues) == 0:
                self.interpolatedInputValues = np.array([np.interp(self.timeOutput, self.timeInput, data)])
            else:
                self.interpolatedInputValues = np.vstack((self.interpolatedInputValues, 
                                                          np.interp(self.timeOutput, self.timeInput, data)))
