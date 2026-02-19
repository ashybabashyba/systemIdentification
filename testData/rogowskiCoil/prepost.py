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


# %% State Space Identfication using Naishadham(2016) and Juang (1997) method

step = 0.01e-9
initialTrainingTime = 0
finalTrainingTime = 20e-9
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

data = np.load("large_gaussian_data.npz")

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
                        systemOutput = system.outputValues, energyThreshold=1-1e-9)

A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

## Plotting initial input and reconstructed training output ##

plt.plot(system.timeOutput*1e9, system.interpolatedInputValues[0], label='Input: Current on Nodal Source')
plt.xlabel('Time (ns)')
plt.ylabel('Current (A)')
# plt.xlim((0, 2e-9))
plt.legend()
plt.grid()
plt.savefig("Rogowski_input_data.png", dpi=300, bbox_inches='tight')
plt.show()


xid, yid = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=system.interpolatedInputValues[0], x0=initialState)

plt.plot(system.timeOutput*1e9, system.outputValues[0]*1e3, color='r', label='Output: Current on Coil')
# plt.plot(system.timeOutput*1e9, yid[0]*1e3, '--', label='Output Reconstructed with SSSI')
plt.xlabel('Time (ns)')
plt.ylabel('Current (mA)')
plt.legend()
plt.grid()
plt.savefig("Rogowski_output_data.png", dpi=300, bbox_inches='tight')
plt.show()


# %% Prediction with the previous parameters

finalTime = np.arange(0, 45e-9 + step, step)
finalOutput = np.interp(
    finalTime,
    data["time_output"],
    data["output_signal"]
).reshape((1, -1))

# input
finalInput = np.interp(
    finalTime,
    data["time_input"],
    data["input_signal"]
).reshape((1, -1))

x_id_predicted, y_id_predicted = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)

plt.plot(finalTime, finalOutput[0], label='Original Output on Coil')
plt.plot(finalTime, y_id_predicted[0], '--', label='SSSI method Output')
plt.axvspan(
    initialTrainingTime,
    finalTrainingTime,
    alpha=0.2,
    label='Training region'
)
plt.xlabel('Time')
plt.ylabel('Current')
plt.legend()
plt.grid()
plt.show()

# %% DTFT for impedance and cutoff frequency estimation

new_freqs = np.geomspace(1e6, 5e8, num=1000)

I_predicted = y_id_predicted[0].squeeze()
t_predicted = finalTime
# I_2_predicted = finalInput[0].squeeze()

## Original output used for training
# I = system.outputValues[0]
# t = system.timeOutput

## Original output used for comparison with prediction
I = finalOutput[0]
t = finalTime
# I_2 = system.interpolatedInputValues[0]

I_f_predicted = np.array([np.sum(I_predicted * np.exp(-1j * 2 * np.pi * f * t_predicted)) for f in new_freqs])
# I_2_f_predicted = np.array([np.sum(I_2_predicted * np.exp(-1j * 2 * np.pi * f * t_predicted)) for f in new_freqs])

I_f = np.array([np.sum(I * np.exp(-1j * 2 * np.pi * f * t)) for f in new_freqs])
# I_2_f = np.array([np.sum(I_2 * np.exp(-1j * 2 * np.pi * f * t)) for f in new_freqs])

# mask_I_2_f_neq_0 = I_2_f_predicted != 0



plt.figure()
plt.plot(new_freqs, np.abs(I_f), '.', label='Original current in frequency domain using DTFT', color='red')
plt.plot(new_freqs, np.abs(I_f_predicted), '.', label='Current from prediction in frequency domain using DTFT', color='blue')
plt.xscale('log')
plt.yscale('log')
# plt.ylim((30, 10e2))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Current [A]')
plt.grid(which='both')
plt.legend()
plt.show()

# plt.figure()
# plt.plot(new_freqs, np.real(I_2_f), '.', label='Current in nodal source, frequency domain using DTFT', color='red')
# plt.xscale('log')
# plt.yscale('log')
# # plt.ylim((30, 10e2))
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Current [A]')
# plt.grid(which='both')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(new_freqs[mask_I_2_f_neq_0], np.real(50*I_f_predicted[mask_I_2_f_neq_0]/I_2_f_predicted[mask_I_2_f_neq_0]), '.', label='Transfer Impedance in frequency domain', color='red')
# plt.xscale('log')
# plt.yscale('log')
# # plt.ylim((1e-1, 1))
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Transfer Impedance [$\Omega$]')
# plt.grid(which='both')
# plt.legend()
# plt.show()


# %%

eigvals = np.linalg.eigvals(A)
plt.plot(np.real(eigvals), np.imag(eigvals), 'o')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Eigenvalues of A matrix')
plt.grid()
plt.axvline(x=0, color='k', linestyle='--')
plt.axhline(y=0, color='k', linestyle='--')
plt.show()
# %% Error vs energy percentages

finalTrainingTimes = np.arange(5e-9, 20e-9, 0.5e-9)
Error = []
energyPercentages = []

data = np.load("large_gaussian_data.npz")

time_input = data["time_input"]
input_signal = data["input_signal"]
time_output_raw = data["time_output"]
output_signal_raw = data["output_signal"]


for finalTrainingTime in finalTrainingTimes:
    step = 0.01e-9
    initialTrainingTime = 0
    newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

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
                            systemOutput = system.outputValues,
                            energyThreshold=1-1e-9)
    
    A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

    finalTime = np.arange(0, 45e-9 + step, step)
    finalOutput = np.interp(
        finalTime,
        data["time_output"],
        data["output_signal"]
    ).reshape((1, -1))

    finalInput = np.interp(
        finalTime,
        data["time_input"],
        data["input_signal"]
    ).reshape((1, -1))

    x_id_predicted, y_id_predicted = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)


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
# %% Generating comparison between methods

## Loading training/prediction data and defining time vectors ##

step = 0.01e-9
initialTrainingTime = 0
finalTrainingTime = 20e-9
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

initialTrainingTime_Naishadham = 0e-9
finalTrainingTime_Naishadham = 20e-9
newTimeVector_Naishadham = np.arange(initialTrainingTime_Naishadham, finalTrainingTime_Naishadham + step, step)


data = np.load("large_gaussian_data.npz")

time_input = data["time_input"]
input_signal = data["input_signal"]
time_output_raw = data["time_output"]
output_signal_raw = data["output_signal"]

## Defining system for Naishadham method ##

system_Naishadham = SystemIdentificationWrapper(
    timeInput=time_input,
    timeOutput=newTimeVector_Naishadham
)

system_Naishadham.addInputData(input_signal)
system_Naishadham.buildInterpolatedInputValues()
system_Naishadham.addOutputData(
    np.interp(newTimeVector_Naishadham, time_output_raw, output_signal_raw)
)

stateSpace_Naishadham = StateSpace(systemInput = system_Naishadham.interpolatedInputValues[0],
                                   systemOutput = system_Naishadham.outputValues,
                                   energyThreshold=1-1e-9,
                                   observabilityMethod='Naishadham')

A_Naishadham, B_Naishadham, C_Naishadham, D_Naishadham, initialState_Naishadham = stateSpace_Naishadham.buildStateSpaceSystem()


## Defining system for Juang method ##

system_Juang = SystemIdentificationWrapper(
    timeInput=time_input,
    timeOutput=newTimeVector
)

system_Juang.addInputData(input_signal)
system_Juang.buildInterpolatedInputValues()
system_Juang.addOutputData(
    np.interp(newTimeVector, time_output_raw, output_signal_raw)
)

stateSpace_Juang = StateSpace(systemInput = system_Juang.interpolatedInputValues[0],
                              systemOutput = system_Juang.outputValues,
                              energyThreshold = 1-1e-9,
                              observabilityMethod = 'Juang')

A_Juang, B_Juang, C_Juang, D_Juang, initialState_Juang = stateSpace_Juang.buildStateSpaceSystem()


## Defining system for Projection method ##

system_Projection = SystemIdentificationWrapper(
    timeInput=time_input,
    timeOutput=newTimeVector)

system_Projection.addInputData(input_signal)
system_Projection.buildInterpolatedInputValues()
system_Projection.addOutputData(
    np.interp(newTimeVector, time_output_raw, output_signal_raw)
)

stateSpace_Projection = StateSpace(systemInput = system_Projection.interpolatedInputValues[0],
                              systemOutput = system_Projection.outputValues,
                              energyThreshold=1-1e-9,
                              observabilityMethod='Projection')

A_Projection, B_Projection, C_Projection, D_Projection, initialState_Projection = stateSpace_Projection.buildStateSpaceSystem()


## Evolving systems  ##

finalTime = np.arange(0, 45e-9 + step, step)
finalOutput = np.interp(
    finalTime,
    data["time_output"],
    data["output_signal"]
).reshape((1, -1))

finalInput = np.interp(
    finalTime,
    data["time_input"],
    data["input_signal"]
).reshape((1, -1))

_, y_id_predicted_Naishadham = stateSpace_Naishadham.evolveInput(A=A_Naishadham, B=B_Naishadham, C=C_Naishadham, D=D_Naishadham, u=finalInput[0], x0=initialState_Naishadham)
_, y_id_predicted_Juang = stateSpace_Juang.evolveInput(A=A_Juang, B=B_Juang, C=C_Juang, D=D_Juang, u=finalInput[0], x0=initialState_Juang)
_, y_id_predicted_Projection = stateSpace_Projection.evolveInput(A=A_Projection, B=B_Projection, C=C_Projection, D=D_Projection, u=finalInput[0], x0=initialState_Projection)


# %%
## Plotting results ##

colors = ['k', 'C0', 'C2', 'C3']
markers = [None, 'o', 's', '^']
labels = ['FDTD simulation Output', 
          'Naishadham method Output', 
          'Juang method Output', 
          'Projection method Output']

fig, axs = plt.subplots(3, 1, figsize=(6, 8))

time_ns = finalTime*1e9 
original = finalOutput[0]*1e3 
naishadham = y_id_predicted_Naishadham[0]*1e3  
juang = y_id_predicted_Juang[0]*1e3  
projection = y_id_predicted_Projection[0]*1e3  

# -------------------------
# 1️⃣ Original vs Naishadham
# -------------------------
axs[0].plot(time_ns, original, color=colors[0], linewidth=2, label=labels[0])
axs[0].plot(time_ns, naishadham, color=colors[1], label=labels[1])
axs[0].grid()
axs[0].axvspan(
    initialTrainingTime*1e9,
    finalTrainingTime*1e9,
    alpha=0.2,
    label='Training region'
)
axs[0].legend()

# -------------------------
# 2️⃣ Original vs Juang
# -------------------------
axs[1].plot(time_ns, original, color=colors[0], linewidth=2, label=labels[0])
axs[1].plot(time_ns, juang, color=colors[2], label=labels[2])
axs[1].grid()
axs[1].axvspan(
    initialTrainingTime*1e9,
    finalTrainingTime*1e9,
    alpha=0.2,
    label='Training region'
)
axs[1].legend()

# -------------------------
# 3️⃣ Original vs Projection
# -------------------------
axs[2].plot(time_ns, original, color=colors[0], linewidth=2, label=labels[0])
axs[2].plot(time_ns, projection, color=colors[3], label=labels[3])
axs[2].grid()
axs[2].axvspan(
    initialTrainingTime*1e9,
    finalTrainingTime*1e9,
    alpha=0.2,
    label='Training region'
)
axs[2].legend()

# Etiquetas globales
for ax in axs.flat:
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Current (mA)")

plt.tight_layout()
plt.savefig("../../../../fig/Rogowski_output_comparison.png", dpi=300, bbox_inches='tight')
plt.show()


# -------------------------
# 4️⃣ Zoom comparison
# -------------------------
plt.plot(time_ns, original, color=colors[0], linewidth=3, label=labels[0])
plt.plot(time_ns, naishadham, '--o', color=colors[1], 
              markersize=8, markevery=23, markerfacecolor='none', 
              label=labels[1])
plt.plot(time_ns, juang, '--s', color=colors[2],
              markersize=8, markevery=25, markerfacecolor='none',
              label=labels[2])
plt.plot(time_ns, projection, '--^', color=colors[3],
              markersize=8, markevery=27, markerfacecolor='none',
              label=labels[3])


plt.xlim(32, 36)
mask = (finalTime >= 32e-9) & (finalTime <= 36e-9)
y_zoom = original[mask]
plt.ylim(min(y_zoom)-0.5, max(y_zoom)+0.5)

plt.grid()
plt.legend()
plt.xlabel('Time (ns)')
plt.ylabel('Current (mA)')


plt.tight_layout()
plt.savefig("../../../../fig/Rogowski_output_comparison_zoom.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
