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


def test_equal_input_output_time_vector():
    timeVector = np.arange(0, 10, 0.1)
    inputSignal = np.sin(timeVector)

    system = SystemIdentificationWrapper(timeInput=timeVector, timeOutput=timeVector)
    system.addInputData(inputSignal)
    system.buildInterpolatedInputValues()

    assert np.allclose(system.interpolatedInputValues[0], inputSignal)

def test_interpolatedInput_output_size_comparison():
    inputTimeVector = np.arange(0, 10, 0.1)
    outputTimeVector = np.arange(0, 10, 0.2)

    inputSignal = np.sin(inputTimeVector)
    outputSignal = np.cos(outputTimeVector)

    system = SystemIdentificationWrapper(timeInput=inputTimeVector, timeOutput=outputTimeVector)

    system.addInputData(inputSignal)
    system.buildInterpolatedInputValues()
    system.addOutputData(outputSignal) 
    
    assert system.interpolatedInputValues.shape[1] == system.outputValues.shape[1]
    assert system.interpolatedInputValues.shape[1] == len(outputTimeVector)

def test_multiple_input_signals():
    inputTimeVector = np.arange(0, 10, 0.1)
    outputTimeVector = np.arange(0, 10, 0.2)

    inputSignal1 = np.sin(inputTimeVector)
    inputSignal2 = np.cos(inputTimeVector)

    system = SystemIdentificationWrapper(timeInput=inputTimeVector, timeOutput=outputTimeVector)
    system.addInputData(inputSignal1)
    system.addInputData(inputSignal2)

    system.buildInterpolatedInputValues()

    assert system.interpolatedInputValues.shape[0] == 2
    assert system.interpolatedInputValues.shape[1] == len(outputTimeVector)