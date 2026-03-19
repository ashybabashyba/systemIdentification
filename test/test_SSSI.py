import numpy as np
import matplotlib.pyplot as plt
import os
import sys

try:
    from src.system_identification_wrapper import SystemIdentificationWrapper
    from src.system_identification import StateSpace
except ImportError:

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

def test_truncated_SVD_construction():
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

    U, S, Vh = stateSpace.buildTruncatedSVD(inputHankelMatrix)

    assert U.shape[1] == len(S) == Vh.shape[0]
    assert np.all(S > stateSpace.truncationThreshold * S[0])

    assert np.allclose(U @ np.diag(S) @ Vh, inputHankelMatrix, atol=1e-6)

    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-6)
    assert np.allclose(Vh @ Vh.T, np.eye(Vh.shape[0]), atol=1e-6)

def test_state_space_construction_and_dimensions():
    inputTimeVector = np.arange(0, 10, 0.1)
    outputTimeVector = np.arange(0, 10, 0.2)
    inputSignal = np.sin(inputTimeVector)
    outputSignal = np.cos(outputTimeVector)

    system = SystemIdentificationWrapper(timeInput=inputTimeVector, timeOutput=outputTimeVector)
    system.addInputData(inputSignal)
    system.buildInterpolatedInputValues()
    system.addOutputData(outputSignal)

    stateSpace = StateSpace(systemInput=system.interpolatedInputValues[0], systemOutput=system.outputValues)
    A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

    assert A.shape[0] == A.shape[1]
    assert B.shape[0] == A.shape[0] and B.shape[1] == 1
    assert C.shape[1] == A.shape[0] and C.shape[0] == 1
    assert D.shape[0] == 1 and D.shape[1] == 1

def test_construction_operators():
    inputTimeVector = np.arange(0, 10, 0.1)
    outputTimeVector = np.arange(0, 10, 0.2)
    inputSignal = np.sin(inputTimeVector)
    outputSignal = np.cos(outputTimeVector)

    system = SystemIdentificationWrapper(timeInput=inputTimeVector, timeOutput=outputTimeVector)
    system.addInputData(inputSignal)
    system.buildInterpolatedInputValues()
    system.addOutputData(outputSignal)

    stateSpace = StateSpace(systemInput=system.interpolatedInputValues[0], systemOutput=system.outputValues)
    inputHankel = stateSpace.buildInputHankelMatrix()

    parallel_operator, orthogonal_operator = stateSpace.buildProjectionOperators(inputHankel)

    assert np.allclose(parallel_operator @ parallel_operator, parallel_operator)
    assert np.allclose(orthogonal_operator @ orthogonal_operator, orthogonal_operator)

    assert np.linalg.norm(parallel_operator @ orthogonal_operator) < 1e-12

def test_orthogonality_operators():
    inputTimeVector = np.arange(0, 10, 0.1)
    outputTimeVector = np.arange(0, 10, 0.2)
    inputSignal = np.sin(inputTimeVector)
    outputSignal = np.cos(outputTimeVector)

    system = SystemIdentificationWrapper(timeInput=inputTimeVector, timeOutput=outputTimeVector)
    system.addInputData(inputSignal)
    system.buildInterpolatedInputValues()
    system.addOutputData(outputSignal)

    stateSpace = StateSpace(systemInput=system.interpolatedInputValues[0], systemOutput=system.outputValues)
    inputHankel = stateSpace.buildInputHankelMatrix()

    parallel_operator, orthogonal_operator = stateSpace.buildProjectionOperators(inputHankel)

    assert np.isclose(np.linalg.norm(parallel_operator @ orthogonal_operator), 0, atol=1e-12)

def test_output_subspace_orthogonality():
    inputTimeVector = np.arange(0, 10, 0.1)
    outputTimeVector = np.arange(0, 10, 0.2)
    inputSignal = np.sin(inputTimeVector)
    outputSignal = np.cos(outputTimeVector)

    system = SystemIdentificationWrapper(timeInput=inputTimeVector, timeOutput=outputTimeVector)
    system.addInputData(inputSignal)
    system.buildInterpolatedInputValues()
    system.addOutputData(outputSignal)

    stateSpace = StateSpace(systemInput=system.interpolatedInputValues[0], systemOutput=system.outputValues)
    outputHankel = stateSpace.buildOutputHankelMatrix()

    parallel_operator, orthogonal_operator = stateSpace.buildProjectionOperators(outputHankel)

    parallel_y = outputHankel @ parallel_operator
    orthogonal_y = outputHankel @ orthogonal_operator

    assert np.isclose(np.linalg.norm(parallel_y) + np.linalg.norm(orthogonal_y), np.linalg.norm(outputHankel), atol=1e-12)
    assert np.isclose(np.linalg.norm(parallel_y @ orthogonal_operator), 0, atol=1e-12)
    assert np.isclose(np.linalg.norm(orthogonal_y @ parallel_operator), 0, atol=1e-12)

def test_input_subspace_orthogonality():
    inputTimeVector = np.arange(0, 10, 0.1)
    outputTimeVector = np.arange(0, 10, 0.2)
    inputSignal = np.sin(inputTimeVector)
    outputSignal = np.cos(outputTimeVector)

    system = SystemIdentificationWrapper(timeInput=inputTimeVector, timeOutput=outputTimeVector)
    system.addInputData(inputSignal)
    system.buildInterpolatedInputValues()
    system.addOutputData(outputSignal)

    stateSpace = StateSpace(systemInput=system.interpolatedInputValues[0], systemOutput=system.outputValues)
    inputHankel = stateSpace.buildInputHankelMatrix()

    parallel_operator, orthogonal_operator = stateSpace.buildProjectionOperators(inputHankel)

    parallel_u = inputHankel @ parallel_operator
    orthogonal_u = inputHankel @ orthogonal_operator

    assert np.isclose(np.linalg.norm(parallel_u) + np.linalg.norm(orthogonal_u), np.linalg.norm(inputHankel), atol=1e-12)
    assert np.isclose(np.linalg.norm(parallel_u @ orthogonal_operator), 0, atol=1e-12)
    assert np.isclose(np.linalg.norm(orthogonal_u @ parallel_operator), 0, atol=1e-12)

def test_RLC_circuit_training_output_reconstruction_and_prediction():
    step = 0.01e-3
    initialTrainingTime = 0
    finalTrainingTime = 1.25e-3
    newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

    trainingFile = "testData/RLC_circuit/RLC_circuit_modulated_gaussian.txt"

    system = SystemIdentificationWrapper(timeInput=np.loadtxt(trainingFile, usecols=0, skiprows=1),
                                        timeOutput=newTimeVector)

    system.addInputData(np.loadtxt(trainingFile, usecols=1, skiprows=1))
    system.buildInterpolatedInputValues()
    system.addOutputData(np.interp(newTimeVector, 
                                np.loadtxt(trainingFile, skiprows=1, usecols=0), 
                                np.loadtxt(trainingFile, skiprows=1, usecols=2)))

    stateSpace = StateSpace(systemInput = system.interpolatedInputValues[0],
                            systemOutput = system.outputValues)

    A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

    _, yid = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=system.interpolatedInputValues[0], x0=initialState)

    assert np.allclose(yid, system.outputValues[0], atol=1e-6)

    predictionFile = "testData/RLC_circuit/RLC_circuit_modulated_gaussian_final.txt"

    finalTime = np.arange(0, 5e-3 + step, step)
    finalOutput = np.interp(finalTime, 
                    np.loadtxt(predictionFile, skiprows=1, usecols=0), 
                    np.loadtxt(predictionFile, skiprows=1, usecols=1)).reshape((1, -1))

    finalInput = np.interp(finalTime, 
                    np.loadtxt(trainingFile, usecols=0, skiprows=1), 
                    np.loadtxt(trainingFile, usecols=1, skiprows=1)).reshape((1, -1))

    _, y_id_predicted = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)

    assert np.allclose(y_id_predicted, finalOutput[0], atol=1e-6)

def test_Rogowski_eigenvalues():
    step = 0.01e-9
    initialTrainingTime = 0
    finalTrainingTime = 20e-9
    newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

    data = np.load("testData/rogowskiCoil/large_gaussian_data.npz")

    time_input = data["time_input"]
    input_signal = data["input_signal"]
    time_output_raw = data["time_output"]
    output_signal_raw = data["output_signal"]

    system = SystemIdentificationWrapper(
        timeInput=time_input,
        timeOutput=newTimeVector
    )

    system.addInputData(input_signal)
    system.buildInterpolatedInputValues()

    system.addOutputData(
        np.interp(newTimeVector, time_output_raw, output_signal_raw)
    )
    stateSpace = StateSpace(systemInput = system.interpolatedInputValues[0],
                            systemOutput = system.outputValues)

    A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

    assert np.all(np.abs(np.linalg.eigvals(A)) < 1)

def test_RLC_multioutput_input_output_comparison():
    step = 0.01e-3
    initialTrainingTime = 0
    finalTrainingTime = 1.0e-3
    newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

    trainingFile = "testData/RLC_circuit/RLC_circuit_modulated_gaussian.txt"
    predictionFile = "testData/RLC_circuit/RLC_circuit_modulated_gaussian_final.txt"

    resistor_voltage_file = "testData/RLC_circuit/RLC_voltage_at_R"
    inductor_voltage_file = "testData/RLC_circuit/RLC_voltage_at_L"
    capacitor_voltage_file = "testData/RLC_circuit/RLC_voltage_at_C"

    system = SystemIdentificationWrapper(timeInput=np.loadtxt(trainingFile, usecols=0, skiprows=1),
                                        timeOutput=newTimeVector)

    system.addInputData(np.loadtxt(trainingFile, usecols=1, skiprows=1))
    system.buildInterpolatedInputValues()

    ## Resistor data ##
    system.addOutputData(np.interp(newTimeVector, 
                                np.loadtxt(trainingFile, skiprows=1, usecols=0), 
                                np.loadtxt(trainingFile, skiprows=1, usecols=2)))
    system.addOutputData(np.interp(newTimeVector, 
                                np.loadtxt(resistor_voltage_file, skiprows=1, usecols=0), 
                                np.loadtxt(resistor_voltage_file, skiprows=1, usecols=1))) 

    ## Inductor data ##
    system.addOutputData(np.interp(newTimeVector, 
                                np.loadtxt(trainingFile, skiprows=1, usecols=0), 
                                np.loadtxt(trainingFile, skiprows=1, usecols=2)))
    system.addOutputData(np.interp(newTimeVector, 
                                np.loadtxt(inductor_voltage_file, skiprows=1, usecols=0), 
                                np.loadtxt(inductor_voltage_file, skiprows=1, usecols=1)))

    ## Capacitor data ##
    system.addOutputData(np.interp(newTimeVector, 
                                np.loadtxt(trainingFile, skiprows=1, usecols=0), 
                                np.loadtxt(trainingFile, skiprows=1, usecols=2)))
    system.addOutputData(np.interp(newTimeVector, 
                                np.loadtxt(capacitor_voltage_file, skiprows=1, usecols=0), 
                                np.loadtxt(capacitor_voltage_file, skiprows=1, usecols=1)))

    stateSpace = StateSpace(systemInput = system.interpolatedInputValues[0],
                            systemOutput = system.outputValues)

    A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

    finalTime = np.arange(0, 5e-3 + step, step)
    finalOutput = np.interp(finalTime, 
                    np.loadtxt(predictionFile, skiprows=1, usecols=0), 
                    np.loadtxt(predictionFile, skiprows=1, usecols=1)).reshape((1, -1))

    finalInput = np.interp(finalTime, 
                    np.loadtxt(trainingFile, usecols=0, skiprows=1), 
                    np.loadtxt(trainingFile, usecols=1, skiprows=1)).reshape((1, -1))

    x_id_predicted, y_id_predicted = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)

    assert np.allclose(-finalInput[0], y_id_predicted[1] + y_id_predicted[3] + y_id_predicted[5], atol=1e-3)
    assert np.allclose(finalOutput[0], y_id_predicted[0], atol=1e-4)
    assert np.allclose(finalOutput[0], y_id_predicted[2], atol=1e-4)
    assert np.allclose(finalOutput[0], y_id_predicted[4], atol=1e-4)

def test_simple_resistor_circuit_component_estimation_and_modification():
    step = 0.01e-3
    initialTrainingTime = 0
    finalTrainingTime = 1.25e-3
    newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

    dataFile = "testData/simple_resistor_circuit/simpleResistor_data"


    system = SystemIdentificationWrapper(timeInput=np.loadtxt(dataFile, usecols=0, skiprows=1),
                                        timeOutput=newTimeVector)

    system.addInputData(-np.loadtxt(dataFile, usecols=1, skiprows=1))
    system.buildInterpolatedInputValues()
    # Current
    system.addOutputData(-np.interp(newTimeVector, 
                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
                                np.loadtxt(dataFile, skiprows=1, usecols=2)))
    # Voltage
    system.addOutputData(-np.interp(newTimeVector, 
                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
                                np.loadtxt(dataFile, skiprows=1, usecols=1)))

    stateSpace = StateSpace(systemInput = system.interpolatedInputValues[0],
                            systemOutput = system.outputValues)

    A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

    assert np.isclose(np.abs(1/D[0]), 50)

    D_modified = D.copy()
    D_modified[0] = D[0] / 2

    finalTime = np.arange(0, 5e-3 + step, step)
    finalInput = np.interp(finalTime, 
                    np.loadtxt(dataFile, usecols=0, skiprows=1), 
                    np.loadtxt(dataFile, usecols=1, skiprows=1)).reshape((1, -1))


    _, y_id = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)
    _, y_id_modified = stateSpace.evolveInput(A=A, B=B, C=C, D=D_modified, u=finalInput[0], x0=initialState)

    assert np.allclose(y_id[0]/2, y_id_modified[0])

def test_simple_resistor_in_series_circuit_component_estimation_and_modification():
    step = 0.01e-3
    initialTrainingTime = 0
    finalTrainingTime = 1.25e-3
    newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

    dataFile = "testData/simple_resistor_circuit/dataSet_seriesR"


    system = SystemIdentificationWrapper(timeInput=np.loadtxt(dataFile, usecols=0, skiprows=1),
                                        timeOutput=newTimeVector)

    system.addInputData(-np.loadtxt(dataFile, usecols=4, skiprows=1))
    system.buildInterpolatedInputValues()
    # Current
    system.addOutputData(-np.interp(newTimeVector, 
                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
                                np.loadtxt(dataFile, skiprows=1, usecols=4)))
    # Voltage at R2
    system.addOutputData(-np.interp(newTimeVector, 
                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
                                np.loadtxt(dataFile, skiprows=1, usecols=1)))

    # Total voltage
    system.addOutputData(-np.interp(newTimeVector, 
                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
                                np.loadtxt(dataFile, skiprows=1, usecols=2)))

    # Voltage at R1
    system.addOutputData(-np.interp(newTimeVector, 
                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
                                np.loadtxt(dataFile, skiprows=1, usecols=3)))

    stateSpace = StateSpace(systemInput = system.interpolatedInputValues[0],
                            systemOutput = system.outputValues)

    A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

    assert np.isclose(np.abs(D[2]), np.abs(D[1] + D[3]))
    assert np.isclose(np.abs(D[1]), 150)
    assert np.isclose(np.abs(D[3]), 50)

    D_modified = D.copy()
    D_modified[3] = 2 * D[3]
    D_modified[2] = np.abs(D_modified[1] + D_modified[3])

    finalTime = np.arange(0, 5e-3 + step, step)
    finalInput = np.interp(finalTime, 
                    np.loadtxt(dataFile, usecols=0, skiprows=1), 
                    np.loadtxt(dataFile, usecols=1, skiprows=1)).reshape((1, -1))

    _, y_id = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)
    _, y_id_modified = stateSpace.evolveInput(A=A, B=B, C=C, D=D_modified, u=finalInput[0], x0=initialState)

    assert np.allclose(2*y_id[3], y_id_modified[3], atol=1e-3)
    assert np.allclose(-y_id_modified[2], y_id_modified[3] + y_id_modified[1], atol=1e-3)

def test_simple_resistor_in_parallel_circuit_component_estimation_and_modification():
    step = 0.01e-3
    initialTrainingTime = 0
    finalTrainingTime = 1.25e-3
    newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

    dataFile = "testData/simple_resistor_circuit/dataSet_parallelR"


    system = SystemIdentificationWrapper(timeInput=np.loadtxt(dataFile, usecols=0, skiprows=1),
                                        timeOutput=newTimeVector)

    system.addInputData(-np.loadtxt(dataFile, usecols=1, skiprows=1))
    system.buildInterpolatedInputValues()
    # Total Current
    system.addOutputData(-np.interp(newTimeVector, 
                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
                                np.loadtxt(dataFile, skiprows=1, usecols=4)))
    # Current at R2
    system.addOutputData(-np.interp(newTimeVector, 
                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
                                np.loadtxt(dataFile, skiprows=1, usecols=3)))

    # Current at R1
    system.addOutputData(-np.interp(newTimeVector, 
                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
                                np.loadtxt(dataFile, skiprows=1, usecols=2)))

    # Voltage
    system.addOutputData(-np.interp(newTimeVector, 
                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
                                np.loadtxt(dataFile, skiprows=1, usecols=1)))

    stateSpace = StateSpace(systemInput = system.interpolatedInputValues[0],
                            systemOutput = system.outputValues)

    A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

    assert np.isclose(np.abs(D[0]), np.abs(D[2] + D[1]))
    assert np.isclose(np.abs(1/D[2]), 100)
    assert np.isclose(np.abs(1/D[1]), 150)

    D_modified = D.copy()
    D_modified[2] = D[2] / 2
    D_modified[0] = np.abs(D_modified[1] + D_modified[2])

    finalTime = np.arange(0, 5e-3 + step, step)
    finalInput = np.interp(finalTime, 
                    np.loadtxt(dataFile, usecols=0, skiprows=1), 
                    np.loadtxt(dataFile, usecols=1, skiprows=1)).reshape((1, -1))
    
    _, y_id = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)
    _, y_id_modified = stateSpace.evolveInput(A=A, B=B, C=C, D=D_modified, u=finalInput[0], x0=initialState)

    assert np.allclose(y_id[2] / 2, y_id_modified[2])
    assert np.allclose(-y_id_modified[0], y_id_modified[1] + y_id_modified[2])