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
## Plot for input-output data of the problem ## 
inputPredictionSignal = np.interp(np.loadtxt(predictionFile, skiprows=1, usecols=0),
                                  np.loadtxt(trainingFile, skiprows=1, usecols=0),
                                  np.loadtxt(trainingFile, usecols=1, skiprows=1))
fig, ax1 = plt.subplots()

line1, = ax1.plot(np.loadtxt(predictionFile, skiprows=1, usecols=0)*1e3,
                  inputPredictionSignal,
                  label='Input: Voltage Source')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Voltage (V)')
ax1.set_ylim(-1, 1)
ax1.grid()

ax2 = ax1.twinx()
line2, = ax2.plot(np.loadtxt(predictionFile, skiprows=1, usecols=0)*1e3,
                  np.loadtxt(predictionFile, skiprows=1, usecols=1)*1e6,
                  '--r',
                  label='Output: Current on Circuit',)
ax2.set_ylabel('Current (µA)')
ax2.grid()
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')
# plt.savefig("../../../../fig/RLC_data.png", dpi=300, bbox_inches='tight')
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

# %% Error vs energy percentages

# finalTrainingTimes = np.array([0.5e-3, 0.75e-3, 1.0e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2.0e-3])
finalTrainingTimes = np.arange(0.75e-3, 2.05e-3, 0.05e-3)
Error = []
energyPercentages = []

trainingFile = "RLC_circuit_modulated_gaussian.txt"
predictionFile = "RLC_circuit_modulated_gaussian_final.txt"

for finalTrainingTime in finalTrainingTimes:
    step = 0.01e-3
    initialTrainingTime = 0
    newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

    system = SystemIdentificationWrapper(
        timeInput=np.loadtxt(trainingFile, usecols=0, skiprows=1),
        timeOutput=newTimeVector
    )

    system.addInputData(np.loadtxt(trainingFile, usecols=1, skiprows=1))
    system.buildInterpolatedInputValues()

    system.addOutputData(
        np.interp(
            newTimeVector,
            np.loadtxt(trainingFile, skiprows=1, usecols=0),
            np.loadtxt(trainingFile, skiprows=1, usecols=2)
        )
    )

    stateSpace = StateSpace(
        systemInput=system.interpolatedInputValues[0],
        systemOutput=system.outputValues,
        energyThreshold=1 - 1e-9
    )

    A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

    finalTime = np.arange(0, 5e-3 + step, step)

    finalOutput = np.interp(
        finalTime,
        np.loadtxt(predictionFile, skiprows=1, usecols=0),
        np.loadtxt(predictionFile, skiprows=1, usecols=1)
    ).reshape((1, -1))

    finalInput = np.interp(
        finalTime,
        np.loadtxt(trainingFile, skiprows=1, usecols=0),
        np.loadtxt(trainingFile, skiprows=1, usecols=1)
    ).reshape((1, -1))

    x_id_predicted, y_id_predicted = stateSpace.evolveInput(
        A=A, B=B, C=C, D=D,
        u=finalInput[0],
        x0=initialState
    )

    dt = step
    N = finalOutput.shape[1]

    window = np.hanning(N)

    Y_true = np.fft.rfft(finalOutput[0] * window)
    Y_pred = np.fft.rfft(y_id_predicted[0] * window)

    Y_true_mag = np.abs(Y_true)
    Y_pred_mag = np.abs(Y_pred)

    error_freq = (
        np.linalg.norm(Y_true_mag - Y_pred_mag) /
        np.max((
            np.linalg.norm(Y_true_mag),
            np.linalg.norm(Y_pred_mag)
        ))
    )

    Error.append(error_freq)

    Y_train = np.fft.rfft(system.outputValues[0] * window[:len(system.outputValues[0])])

    energy_training = np.sum(np.abs(Y_train)**2)
    energy_total = np.sum(np.abs(Y_true)**2)

    energyPercentage = energy_training / energy_total * 100
    energyPercentages.append(energyPercentage)


plt.plot(energyPercentages, Error, marker='o')
plt.xlabel('Energy Percentage (Frequency Domain) [%]')
plt.ylabel('Relative Spectral Error')
plt.title('Spectral Error vs Energy Percentage')
plt.grid()
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

# %% Generating comparison between methods

## Loading training/prediction data and defining time vectors ##

step = 0.01e-3
initialTrainingTime = 0
finalTrainingTime = 1.25e-3
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

initialTrainingTime_Naishadham = 1.8e-3
finalTrainingTime_Naishadham = 3e-3
newTimeVector_Naishadham = np.arange(initialTrainingTime_Naishadham, finalTrainingTime_Naishadham + step, step)


trainingFile = "RLC_circuit_modulated_gaussian.txt"
predictionFile = "RLC_circuit_modulated_gaussian_final.txt"


## Defining system for Naishadham method ##

system_Naishadham = SystemIdentificationWrapper(timeInput=np.loadtxt(trainingFile, usecols=0, skiprows=1),
                                     timeOutput=newTimeVector_Naishadham)

system_Naishadham.addInputData(np.loadtxt(trainingFile, usecols=1, skiprows=1))
system_Naishadham.buildInterpolatedInputValues()
system_Naishadham.addOutputData(np.interp(newTimeVector_Naishadham, 
                               np.loadtxt(trainingFile, skiprows=1, usecols=0), 
                               np.loadtxt(trainingFile, skiprows=1, usecols=2)))

stateSpace_Naishadham = StateSpace(systemInput = system_Naishadham.interpolatedInputValues[0],
                                   systemOutput = system_Naishadham.outputValues,
                                   energyThreshold=1-1e-15,
                                   observabilityMethod='Naishadham')

A_Naishadham, B_Naishadham, C_Naishadham, D_Naishadham, initialState_Naishadham = stateSpace_Naishadham.buildStateSpaceSystem()


## Defining system for Juang method ##

system_Juang = SystemIdentificationWrapper(timeInput=np.loadtxt(trainingFile, usecols=0, skiprows=1),
                                        timeOutput=newTimeVector)

system_Juang.addInputData(np.loadtxt(trainingFile, usecols=1, skiprows=1))
system_Juang.buildInterpolatedInputValues()
system_Juang.addOutputData(np.interp(newTimeVector, 
                               np.loadtxt(trainingFile, skiprows=1, usecols=0), 
                               np.loadtxt(trainingFile, skiprows=1, usecols=2)))

stateSpace_Juang = StateSpace(systemInput = system_Juang.interpolatedInputValues[0],
                              systemOutput = system_Juang.outputValues,
                              energyThreshold=1-1e-6,
                              observabilityMethod='Juang')

A_Juang, B_Juang, C_Juang, D_Juang, initialState_Juang = stateSpace_Juang.buildStateSpaceSystem()


## Defining system for Projection method ##

system_Projection = SystemIdentificationWrapper(timeInput=np.loadtxt(trainingFile, usecols=0, skiprows=1),
                                        timeOutput=newTimeVector)

system_Projection.addInputData(np.loadtxt(trainingFile, usecols=1, skiprows=1))
system_Projection.buildInterpolatedInputValues()
system_Projection.addOutputData(np.interp(newTimeVector, 
                               np.loadtxt(trainingFile, skiprows=1, usecols=0), 
                               np.loadtxt(trainingFile, skiprows=1, usecols=2)))

stateSpace_Projection = StateSpace(systemInput = system_Projection.interpolatedInputValues[0],
                              systemOutput = system_Projection.outputValues,
                              energyThreshold=1-1e-6,
                              observabilityMethod='Projection')

A_Projection, B_Projection, C_Projection, D_Projection, initialState_Projection = stateSpace_Projection.buildStateSpaceSystem()


## Evolving systems  ##

finalTime = np.arange(0, 5e-3 + step, step)
finalOutput = np.interp(finalTime, 
                   np.loadtxt(predictionFile, skiprows=1, usecols=0), 
                   np.loadtxt(predictionFile, skiprows=1, usecols=1)).reshape((1, -1))

finalInput = np.interp(finalTime, 
                   np.loadtxt(trainingFile, usecols=0, skiprows=1), 
                   np.loadtxt(trainingFile, usecols=1, skiprows=1)).reshape((1, -1))

_, y_id_predicted_Naishadham = stateSpace_Naishadham.evolveInput(A=A_Naishadham, B=B_Naishadham, C=C_Naishadham, D=D_Naishadham, u=finalInput[0], x0=initialState_Naishadham)
_, y_id_predicted_Juang = stateSpace_Juang.evolveInput(A=A_Juang, B=B_Juang, C=C_Juang, D=D_Juang, u=finalInput[0], x0=initialState_Juang)
_, y_id_predicted_Projection = stateSpace_Projection.evolveInput(A=A_Projection, B=B_Projection, C=C_Projection, D=D_Projection, u=finalInput[0], x0=initialState_Projection)


## Plotting results ##

fig, axs = plt.subplots(1, 1, figsize=(10, 8))

# Colores y marcadores consistentes
colors = ['k', 'C0', 'C2', 'C3']
markers = [None, 'o', 's', '^']
labels = ['Original Output', 'Naishadham\'s method', 
          'Juang\'s method', 'Projection method [this work]']

fontsize_axes = 14     # números de los ejes
fontsize_labels = 16   # xlabel / ylabel
fontsize_legend = 14   # legend

# Subplot 0: µA
axs.plot(finalTime*1e3, finalOutput[0]*1e6, color=colors[0], linewidth=3, label=labels[0])
axs.plot(finalTime*1e3, y_id_predicted_Naishadham[0]*1e6, '--'+markers[1], 
            color=colors[1], markersize=6, markevery=15, markerfacecolor='none', label=labels[1])
axs.plot(finalTime*1e3, y_id_predicted_Juang[0]*1e6, '--'+markers[2], 
            color=colors[2], markersize=6, markevery=15, markerfacecolor='none', label=labels[2])
axs.plot(finalTime*1e3, y_id_predicted_Projection[0]*1e6, '--'+markers[3], 
            color=colors[3], markersize=6, markevery=15, markerfacecolor='none', label=labels[3])

axs.axvspan(initialTrainingTime*1e3, finalTrainingTime*1e3, alpha=0.3)
axs.set_xlabel('Time (ms)', fontsize=fontsize_labels)
axs.set_ylabel('Current (µA)', fontsize=fontsize_labels)
axs.tick_params(axis='both', labelsize=fontsize_axes)
# axs[0].set_xlim(0, 3)
axs.grid()
# axs[0].legend(fontsize=fontsize_legend)
axs.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=fontsize_legend)

# # Subplot 1: nA zoom
# axs[1].plot(finalTime*1e3, finalOutput[0]*1e9, color=colors[0], linewidth=2, label=labels[0])
# axs[1].plot(finalTime*1e3, y_id_predicted_Naishadham[0]*1e9, '--'+markers[1], 
#             color=colors[1], markersize=6, markevery=15, markerfacecolor='none', label=labels[1])
# axs[1].plot(finalTime*1e3, y_id_predicted_Juang[0]*1e9, '--'+markers[2], 
#             color=colors[2], markersize=6, markevery=15, markerfacecolor='none', label=labels[2])
# axs[1].plot(finalTime*1e3, y_id_predicted_Projection[0]*1e9, '--'+markers[3], 
#             color=colors[3], markersize=6, markevery=15, markerfacecolor='none', label=labels[3])

# axs[1].set_xlim(2, 3)
# axs[1].set_ylim(-50, 50)
# axs[1].set_xlabel('Time (ms)', fontsize=fontsize_labels)
# axs[1].set_ylabel('Current (nA)', fontsize=fontsize_labels)
# axs[1].tick_params(axis='both', labelsize=fontsize_axes)
# axs[1].grid()

plt.tight_layout()
# plt.savefig("../../../../fig/RLC_output_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
