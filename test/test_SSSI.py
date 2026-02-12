import numpy as np
import matplotlib.pyplot as plt

try:
    from src.system_identification_wrapper import SystemIdentificationWrapper
    from src.system_identification import StateSpace
except ImportError:
    import os
    import sys

    sys.path.append(os.pardir)
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../', 'src'))
    from system_identification_wrapper import SystemIdentificationWrapper
    from system_identification import StateSpace


def test_input_Hankel_matrix_construction():
    inputTimeVector = np.arange(0, 10, 0.1)
    outputTimeVector = np.arange(0, 10, 0.2)
    inputSignal = np.sin(inputTimeVector)
    outputSignal = np.cos(outputTimeVector)

    system = SystemIdentificationWrapper(timeInput=inputTimeVector, timeOutput=outputTimeVector)
    system.addInputData(inputSignal)
    system.buildInterpolatedInputValues()
    system.addOutputData(outputSignal)

    stateSpace = StateSpace(systemInput=system.interpolatedInputValues[0], systemOutput=system.outputValues)

    inputHankelMatrix = stateSpace.buildInputHankelMatrix()

    for i in range(inputHankelMatrix.shape[1] - 1):
        assert np.allclose(inputHankelMatrix[:-1, i+1], inputHankelMatrix[1:, i])


def test_output_Hankel_matrix_construction(): 
    inputTimeVector = np.arange(0, 10, 0.1)
    outputTimeVector = np.arange(0, 10, 0.2)
    inputSignal = np.sin(inputTimeVector)
    outputSignal = np.cos(outputTimeVector)

    system = SystemIdentificationWrapper(timeInput=inputTimeVector, timeOutput=outputTimeVector)
    system.addInputData(inputSignal)
    system.buildInterpolatedInputValues()
    system.addOutputData(outputSignal)

    stateSpace = StateSpace(systemInput=system.interpolatedInputValues[0], systemOutput=system.outputValues)

    outputHankelMatrix = stateSpace.buildOutputHankelMatrix()

    for i in range(outputHankelMatrix.shape[1] - 1):
        assert np.allclose(outputHankelMatrix[:-1, i+1], outputHankelMatrix[1:, i])