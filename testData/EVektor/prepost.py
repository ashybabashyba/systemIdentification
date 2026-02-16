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
step = 0.005e-6
initialTrainingTime = 0
finalTrainingTime = 12.0e-6
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

# data = np.load("nodal_double_gaussian_board_top.npz")
data = np.load("nodal_double_gaussian_board_top_MinorResistance.npz")

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
                        systemOutput = system.outputValues, energyThreshold=1-1e-6)

A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

## Plotting initial input and reconstructed training output ##

plt.plot(system.timeOutput, system.interpolatedInputValues[0], label='Input Current')
plt.xlabel('Time')
plt.ylabel('Current')
# plt.xlim((0, 2e-10))
plt.legend()
plt.grid()
plt.show()


xid, yid = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=system.interpolatedInputValues[0], x0=initialState)
plt.plot(system.timeOutput, system.outputValues[0], label='Original Output')
plt.plot(system.timeOutput, yid[0], '--', label='Output Reconstructed with SSSI')
plt.xlabel('Time')
plt.ylabel('Current')
plt.legend()
plt.grid()
plt.show()
# %%

finalTime = np.arange(0, 50e-6 + step, step)
finalOutput = np.interp(
    finalTime,
    time_output_raw,
    output_signal_raw
).reshape((1, -1))

# input
finalInput = np.interp(
    finalTime,
    time_input,
    input_signal
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


# %%

segment_duration = 10e-6
subplot_duration = 2.5e-6
num_segments = 5

time = finalTime
y_true = finalOutput[0]
y_pred = y_id_predicted[0]

for k in range(num_segments):

    segment_start = k * segment_duration
    segment_end = (k + 1) * segment_duration

    fig, axes = plt.subplots(2, 2, figsize=(10, 5), constrained_layout=True)
    axes = axes.flatten()

    for i in range(4):
        sub_start = segment_start + i * subplot_duration
        sub_end = sub_start + subplot_duration

        mask = (time >= sub_start) & (time <= sub_end)

        ax = axes[i]
        ax.plot(time[mask], y_true[mask], label='Original Output')
        ax.plot(time[mask], y_pred[mask], '--', label='SSSI Output')

        ax.set_title(
            f'{sub_start*1e6:.2f} – {sub_end*1e6:.2f} µs',
            fontsize=10
        )
        ax.grid(True)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Current [A]')

        if i == 0:
            ax.legend(fontsize=9)


    fig.suptitle(
        f'Segment {k+1}: {segment_start*1e6:.1f} – {segment_end*1e6:.1f} $\mu$ s',
        fontsize=12
    )

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

# %%
