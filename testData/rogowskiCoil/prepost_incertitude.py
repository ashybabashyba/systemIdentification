# %%
import numpy as np
import matplotlib.pyplot as plt


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
finalControlTime = 25e-9
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

std_lower = np.clip(y_kalman[output_index] - observer.outputStd[output_index], ylim_min, ylim_max)
std_upper = np.clip(y_kalman[output_index] + observer.outputStd[output_index], ylim_min, ylim_max)

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
# plt.xlim(20e-9, 30e-9)
# plt.ylim(np.min(finalOutput[output_index])*3.5, np.max(finalOutput[output_index])*3.5)
plt.legend()
plt.grid()
plt.show()

# %%
