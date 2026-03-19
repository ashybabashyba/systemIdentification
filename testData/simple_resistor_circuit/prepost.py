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

# %% 
# Now let's try to change the resistor value in D, for example, if we have the double of the resistance, then D[0] should be halved,
# similarly, the final current should be halved

D_modified = D.copy()
D_modified[0] = D[0] / 2

finalTime = np.arange(0, 5e-3 + step, step)
finalOutput = np.interp(finalTime, 
                   np.loadtxt(dataFile, skiprows=1, usecols=0), 
                   np.loadtxt(dataFile, skiprows=1, usecols=2)).reshape((1, -1))

finalInput = np.interp(finalTime, 
                   np.loadtxt(dataFile, usecols=0, skiprows=1), 
                   np.loadtxt(dataFile, usecols=1, skiprows=1)).reshape((1, -1))


#Evolving the system with D not modified
_, y_id = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)

# Evolving the system with D modified
_, y_id_modified = stateSpace.evolveInput(A=A, B=B, C=C, D=D_modified, u=finalInput[0], x0=initialState)

plt.plot(finalTime, y_id[0]/2, label='Reconstructed current halved')
plt.plot(finalTime, y_id_modified[0], '--', label='Reconstructed current with modified D')
plt.xlabel('Time')
plt.ylabel('Current')
plt.legend()
plt.grid()
plt.show()



# %% -------------------------------------------------------------------------------------------------------------------------------------
# %% Resistance estimation for two resistors in series R1=50 and R2=150

step = 0.01e-3
initialTrainingTime = 0
finalTrainingTime = 1.25e-3
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

dataFile = "dataSet_seriesR"


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

print("Value estimated for the equivalent resistor:", D[2], "Ohms") # I = Vs/Req
print("Value estimated for R1 =", D[3], "Ohms")
print("Value estimated for R2 =", D[1], "Ohms")

# %% 
# Now let's try to change one of the resistors value in D, doubling R1 and also changing the total resistance

D_modified = D.copy()
D_modified[3] = 2 * D[3]
D_modified[2] = np.abs(D_modified[1] + D_modified[3])

finalTime = np.arange(0, 5e-3 + step, step)
finalOutput = np.interp(finalTime, 
                   np.loadtxt(dataFile, skiprows=1, usecols=0), 
                   np.loadtxt(dataFile, skiprows=1, usecols=2)).reshape((1, -1))

finalInput = np.interp(finalTime, 
                   np.loadtxt(dataFile, usecols=0, skiprows=1), 
                   np.loadtxt(dataFile, usecols=1, skiprows=1)).reshape((1, -1))


#Evolving the system with D not modified
_, y_id = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)

# Evolving the system with D modified
_, y_id_modified = stateSpace.evolveInput(A=A, B=B, C=C, D=D_modified, u=finalInput[0], x0=initialState)

plt.plot(finalTime, y_id[3]*2, label='Reconstructed voltage at R1 halved')
plt.plot(finalTime, y_id_modified[3], '--', label='Reconstructed voltage at R1 with modified D')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()
plt.grid()
plt.show()


# %% -------------------------------------------------------------------------------------------------------------------------------------
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
