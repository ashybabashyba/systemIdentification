# %%
import numpy as np
import matplotlib.pyplot as plt

try:
    from sippy_unipi import system_identification
    from src.system_identification_wrapper import SystemIdentificationWrapper
except ImportError:
    import os
    import sys

    sys.path.append(os.pardir)
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../', 'src'))
    from sippy_unipi import system_identification
    from system_identification_wrapper import SystemIdentificationWrapper
    from system_identification import StateSpace

from sippy_unipi import functionset as fset
from sippy_unipi import functionsetSIM as fsetSIM

# %%
step = 0.01e-3
initialTrainingTime = 0
finalTrainingTime = 1.25e-3
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

dataFile = "simpleResistor_data"


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

print("Value estimated for the resistor:", 1/D[0], "Ohms")

# %% Resistance estimation for two resistors in series R1=50 and R2=150

step = 0.01e-3
initialTrainingTime = 0
finalTrainingTime = 1.25e-3
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

dataFile = "dataSet_seriesR"


system = SystemIdentificationWrapper(timeInput=np.loadtxt(dataFile, usecols=0, skiprows=1),
                                     timeOutput=newTimeVector)

system.addInputData(-np.loadtxt(dataFile, usecols=2, skiprows=1))
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

print("Value estimated for the equivalent resistor:", 1/D[0], "Ohms") # I = Vs/Req
print("Value estimated for R1/Req:", D[3], "then R1 =", D[3]/D[0], "Ohms")
print("Value estimated for R2/Req:", D[1], "then R2 =", D[1]/D[0], "Ohms")


# %% Resistance estimation for two resistors in parallel R1=100 and R2=150

step = 0.01e-3
initialTrainingTime = 0
finalTrainingTime = 1.25e-3
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

dataFile = "dataSet_parallelR"


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

print("Value estimated for the equivalent resistor:", 1/D[0], "Ohms") # I = Vs/Req
print("Value estimated for R1:", 1/D[2], "Ohms")
print("Value estimated for R2:", 1/D[1], "Ohms")
# %%
