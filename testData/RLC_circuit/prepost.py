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

# trainingFile = "RLC_circuit_ramp.txt"
# predictionFile = "RLC_circuit_ramp_final.txt"

trainingFile = "RLC_circuit_modulated_gaussian.txt"
predictionFile = "RLC_circuit_modulated_gaussian_final.txt"

system = SystemIdentificationWrapper(timeInput=np.loadtxt(trainingFile, usecols=0, skiprows=1),
                                     timeOutput=newTimeVector)

system.addInputData(np.loadtxt(trainingFile, usecols=1, skiprows=1))
system.buildInterpolatedInputValues()
system.addOutputData(np.interp(newTimeVector, 
                               np.loadtxt(trainingFile, skiprows=1, usecols=0), 
                               np.loadtxt(trainingFile, skiprows=1, usecols=2)))

stateSpace = StateSpace(systemInput = system.interpolatedInputValues[0],
                        systemOutput = system.outputValues,
                        energyThreshold=1-1e-15)

A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

## Plotting initial input and reconstructed training output ##

plt.plot(system.timeInput*1e3, system.inputValues[0], label='Input Voltage')
plt.xlabel('Time [ms]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid()
plt.show()

xid, yid = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=system.interpolatedInputValues[0], x0=initialState)
plt.plot(system.timeOutput*1e3, system.outputValues[0]*1e6, label='Original Output')
plt.plot(system.timeOutput*1e3, yid[0]*1e6, '--', label='Output Reconstructed with SSSI')
plt.xlabel('Time [ms]')
plt.ylabel('Current [$\mu$A]')
plt.legend()
plt.grid()
plt.show()
# %%

finalTime = np.arange(0, 5e-3 + step, step)
finalOutput = np.interp(finalTime, 
                   np.loadtxt(predictionFile, skiprows=1, usecols=0), 
                   np.loadtxt(predictionFile, skiprows=1, usecols=1)).reshape((1, -1))

finalInput = np.interp(finalTime, 
                   np.loadtxt(trainingFile, usecols=0, skiprows=1), 
                   np.loadtxt(trainingFile, usecols=1, skiprows=1)).reshape((1, -1))

x_id_predicted, y_id_predicted = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(finalTime, finalOutput[0], label='Original Output')
axs[0].plot(finalTime, y_id_predicted[0], '--', label='SSSI method Output')
axs[0].axvspan(
    initialTrainingTime,
    finalTrainingTime,
    alpha=0.2,
    label='Training region'
)
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Current')
axs[0].legend()
axs[0].grid()
axs[0].set_title('Full signal')

axs[1].plot(finalTime, finalOutput[0], label='Original Output')
axs[1].plot(finalTime, y_id_predicted[0], '--', label='SSSI method Output')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Current')
axs[1].set_xlim(2e-3, 5e-3)
axs[1].set_ylim(-2.5e-8, 2.5e-8)
axs[1].grid()
axs[1].set_title('Zoom between 2ms and 5ms')

plt.tight_layout()
plt.show()


# %% Defining a modulated gaussian pulse input

t = np.linspace(0, 5e-3, 2000)
t0 = 1e-3
sigma = 3e-4
f0 = 3e3
signal = np.exp(-(t - t0)**2 / (2 * sigma**2)) * np.cos(2 * np.pi * f0 * (t - t0))

plt.plot(t*1e3, signal)
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude')
plt.title('Modulated Gaussian Pulse')
plt.grid()
plt.show()

# np.savetxt("RLC_circuit_modulated_gaussian_pulse_input.txt", np.column_stack((t, signal)))

# %%
