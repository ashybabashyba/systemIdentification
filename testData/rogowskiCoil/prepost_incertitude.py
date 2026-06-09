# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal


import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from src.system_identification_wrapper import SystemIdentificationWrapper
from src.system_identification import StateSpace

from src.filter_identification_wrapper import FilterIdentificationWrapper
from src.ok_filter_identification import ObserverKalmanFilterIdentification



# %%

step = 0.01e-9
initialTrainingTime = 0
finalTrainingTime = 20e-9
finalControlTime = 30e-9
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

data = np.load("large_gaussian_data.npz")

time_input = data["time_input"]
input_signal = data["input_signal"]
time_output_raw = data["time_output"]
output_signal_raw = data["output_signal"]

# %%

## ------------ OPTIONAL: NUMERICAL NOISE ------------ ##

noise_level = 0.5  
sigma_signal = np.std(output_signal_raw)
sigma_noise = noise_level * sigma_signal

noise = np.random.normal(loc=0.0, scale=sigma_noise, size=output_signal_raw.shape)

output_signal_noisy = output_signal_raw + noise

## ------------ OPTIONAL: NUMERICAL NOISE ------------ ##

## ------------ Filter for noisy data ------------ ##
dt = newTimeVector[1] - newTimeVector[0]
fs = 1.0 / dt

fft_vals = np.abs(np.fft.rfft(output_signal_raw))
fft_freqs = np.fft.rfftfreq(len(output_signal_raw), d=dt)

umbral_física = 0.01 * np.max(fft_vals)
indices_física = np.where(fft_vals > umbral_física)[0]

if len(indices_física) > 0:
    f_max_física = fft_freqs[indices_física[-1]]  
    fc = f_max_física * 1.25  
else:
    fc = fs * 0.1  

f_nyquist = 0.5 * fs
Wn = fc / f_nyquist  

b, a = signal.butter(4, Wn, btype='low')

input_signal = signal.filtfilt(b, a, input_signal)
output_signal_noisy = signal.filtfilt(b, a, output_signal_noisy)

# %%

system = SystemIdentificationWrapper(
    timeInput=time_input,
    timeOutput=newTimeVector
)

system.addInputData(input_signal)
system.buildInterpolatedInputValues()

# system.addOutputData(
#     np.interp(newTimeVector, time_output_raw, output_signal_raw)
# )
system.addOutputData(
    np.interp(newTimeVector, time_output_raw, output_signal_noisy)
)

stateSpace = StateSpace(systemInput = system.interpolatedInputValues[0],
                        systemOutput = system.outputValues)

A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

# %%

plt.plot(time_output_raw*1e9, output_signal_raw, 'b', label='Output without noise')
plt.plot(time_output_raw*1e9, output_signal_noisy, '--r', label='Output with noise')
plt.xlabel('Time (ns)')
plt.ylabel('Current (A)')
plt.xlim((0, 20))
# plt.ylim((-0.005, 0.005))
plt.legend()
plt.grid()
plt.show()

# %%

finalTime = np.arange(0, 45e-9 + step, step)
# finalOutput = np.interp(
#     finalTime,
#     data["time_output"],
#     data["output_signal"]
# ).reshape((1, -1))

finalOutput = np.interp(
    finalTime,
    data["time_output"],
    output_signal_noisy
).reshape((1, -1))

# input
finalInput = np.interp(
    finalTime,
    data["time_input"],
    data["input_signal"]
).reshape((1, -1))

x_id_predicted, y_id_predicted = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)

data_wrapper = FilterIdentificationWrapper()
data_wrapper.addInputData(finalInput[0], finalTime)
data_wrapper.addReferenceOutput(finalOutput[0], finalTime)  
data_wrapper.addDeterministicOutput(y_id_predicted[0], finalTime)

data_wrapper.setTimeWindow(t_training=finalTrainingTime, t_control=finalControlTime) 


observer = ObserverKalmanFilterIdentification(
    A=A, B=B, C=C, D=D, 
    initialState=initialState, 
    dataWrapper=data_wrapper,
    energyThreshold=1-1e-9
)
x_kalman, y_kalman = observer.evolveWithFilter(x_ref=x_id_predicted.T)

# %%

sigma_y = observer.computeUncertaintyWorstCase()

# %%

output_index = 0

ylim_min, ylim_max = -0.007, 0.017

plt.plot(finalTime,    finalOutput[output_index], label='Original output', linewidth=2, color='black')
plt.plot(finalTime, y_id_predicted[output_index], '-.', label='Predicted with SSSI')
plt.plot(finalTime,       y_kalman[output_index], '--', label='Reconstructed with observer correction')

std_lower = np.clip(y_id_predicted[output_index] - observer.outputStd[output_index], ylim_min, ylim_max)
std_upper = np.clip(y_id_predicted[output_index] + observer.outputStd[output_index], ylim_min, ylim_max)

plt.fill_between(
    finalTime,
    std_lower,
    std_upper,
    alpha=0.5,
    label=f'Uncertainty (max std: {np.max(observer.outputStd[output_index][1:]):.2e})'
)
plt.vlines(x=[initialTrainingTime, finalTrainingTime], ymin=ylim_min, ymax=ylim_max, colors='gray', linestyles='--', label='Training interval')
plt.vlines(x=[finalControlTime], ymin=ylim_min, ymax=ylim_max, colors='magenta', linestyles='--', label='Control interval end')
plt.xlabel('Time')
plt.ylim(ylim_min, ylim_max)
plt.xlim(0e-9, 30e-9)
# plt.ylim(np.min(finalOutput[output_index])*3.5, np.max(finalOutput[output_index])*3.5)
plt.legend()
plt.grid()
plt.show()

# %%

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

xlims = [
    (0, 10e-9),
    (10e-9, 20e-9),
    (20e-9, 30e-9),
    (30e-9, 40e-9),
    (40e-9, 45e-9),
    (0, 45e-9)
]

std_lower = np.clip(y_id_predicted[output_index] - observer.outputStd[output_index], ylim_min, ylim_max)
std_upper = np.clip(y_id_predicted[output_index] + observer.outputStd[output_index], ylim_min, ylim_max)

for i, ax in enumerate(axes):
    ax.plot(finalTime, finalOutput[output_index], label='Original output', linewidth=2, color='black')
    ax.plot(finalTime, y_id_predicted[output_index], '-.', label='Predicted with SSSI')
    ax.plot(finalTime, y_kalman[output_index], '--', label='Reconstructed with observer correction')
    
    ax.fill_between(
        finalTime,
        std_lower,
        std_upper,
        alpha=0.5,
        label=f'Uncertainty (max std: {np.max(observer.outputStd[output_index][1:]):.2e})'
    )
    ax.vlines(x=[initialTrainingTime, finalTrainingTime], ymin=ylim_min, ymax=ylim_max, colors='gray', linestyles='--', label='Training interval')
    ax.vlines(x=[finalControlTime], ymin=ylim_min, ymax=ylim_max, colors='magenta', linestyles='--', label='Control interval end')
    
    ax.set_xlim(xlims[i])
    ax.set_ylim(ylim_min, ylim_max)
    ax.grid(True)

fig.text(0.5, 0.02, 'Time', ha='center')
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)

plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.show()

# %%
