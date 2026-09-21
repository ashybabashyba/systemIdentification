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

from src.evt_statistical_bound import EVTStatisticalBound


# %%

step = 0.01e-9
initialTrainingTime = 0
finalTrainingTime = 24e-9
finalControlTime = finalTrainingTime + 10e-9
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

data = np.load("large_gaussian_data.npz")

time_input = data["time_input"]
input_signal = data["input_signal"]
time_output_raw = data["time_output"]
output_signal_raw = data["output_signal"]


# %%

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


# %%

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

# %%
control_TimeVector = np.arange(initialTrainingTime, finalControlTime + step, step)
referenceOutput_control = np.interp(
    control_TimeVector,
    data["time_output"],
    data["output_signal"]
).reshape((1, -1))
predictedOutput_control = np.interp(
    control_TimeVector,
    finalTime,
    y_id_predicted[0]
).reshape((1, -1))

residuals = referenceOutput_control[0] - predictedOutput_control[0]
evt_bound = EVTStatisticalBound(
    residuals=residuals,
    number_of_pending_iterations=len(finalTime)-len(control_TimeVector),
    confidence_level=0.99
)

B = evt_bound.compute_statistical_bound()


# %%

output_index = 0

ylim_min, ylim_max = -0.007, 0.017
uncertainty_mask = finalTime > finalControlTime

plt.plot(finalTime,    finalOutput[output_index], label='Original output', linewidth=2, color='black')
plt.plot(finalTime, y_id_predicted[output_index], '-.', label='Predicted with SSSI')
plt.fill_between(
    finalTime[uncertainty_mask],
    y_id_predicted[output_index, uncertainty_mask] - B,
    y_id_predicted[output_index, uncertainty_mask] + B,
    alpha=0.3,
    label=f'EVT uncertainty bound {B:.2e}'
)

plt.vlines(x=[initialTrainingTime, finalTrainingTime], ymin=ylim_min, ymax=ylim_max, colors='gray', linestyles='--', label='Training interval')
plt.vlines(x=[finalControlTime], ymin=ylim_min, ymax=ylim_max, colors='magenta', linestyles='--', label='Control interval end')
plt.xlabel('Time')
plt.ylim(ylim_min, ylim_max)
# plt.xlim(30e-9, 40e-9)
# plt.ylim(np.min(finalOutput[output_index])*3.5, np.max(finalOutput[output_index])*3.5)
plt.legend()
plt.grid()
plt.show()

# %%

# fig, axes = plt.subplots(3, 2, figsize=(12, 10))
# axes = axes.flatten()

# xlims = [
#     (0, 10e-9),
#     (10e-9, 20e-9),
#     (20e-9, 30e-9),
#     (30e-9, 40e-9),
#     (40e-9, 45e-9),
#     (0, 45e-9)
# ]

# std_lower = np.clip(y_id_predicted[output_index] - observer.outputStd[output_index], ylim_min, ylim_max)
# std_upper = np.clip(y_id_predicted[output_index] + observer.outputStd[output_index], ylim_min, ylim_max)

# for i, ax in enumerate(axes):
#     ax.plot(finalTime, finalOutput[output_index], label='Original output', linewidth=2, color='black')
#     ax.plot(finalTime, y_id_predicted[output_index], '-.', label='Predicted with SSSI')
#     ax.plot(finalTime, y_kalman[output_index], '--', label='Reconstructed with observer correction')
    
#     ax.fill_between(
#         finalTime,
#         std_lower,
#         std_upper,
#         alpha=0.5,
#         label=f'Uncertainty (max std: {np.max(observer.outputStd[output_index][1:]):.2e})'
#     )
#     ax.vlines(x=[initialTrainingTime, finalTrainingTime], ymin=ylim_min, ymax=ylim_max, colors='gray', linestyles='--', label='Training interval')
#     ax.vlines(x=[finalControlTime], ymin=ylim_min, ymax=ylim_max, colors='magenta', linestyles='--', label='Control interval end')
    
#     ax.set_xlim(xlims[i])
#     ax.set_ylim(ylim_min, ylim_max)
#     ax.grid(True)

# fig.text(0.5, 0.02, 'Time', ha='center')
# handles, labels = axes[0].get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# fig.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)

# plt.tight_layout(rect=[0, 0.05, 1, 0.93])
# plt.show()

# %%
