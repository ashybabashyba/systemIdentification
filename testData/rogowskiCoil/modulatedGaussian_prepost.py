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

# %% Using Modulated Gaussian Excitation for Identification

modulated_data = np.load("modulated_gaussian_data.npz")

modulated_time_input = modulated_data["time_input"]
modulated_input_signal = modulated_data["input_signal"]
modulated_time_output = modulated_data["time_output"]
modulated_output_signal = modulated_data["output_signal"]

new_step = 0.01e-9
initialTrainingTime = 0
finalTrainingTime = 35e-9
new_time = np.arange(initialTrainingTime, finalTrainingTime + new_step, new_step)

new_system = SystemIdentificationWrapper(
    timeInput=modulated_time_input,
    timeOutput=new_time
)

new_system.addInputData(modulated_input_signal)
new_system.buildInterpolatedInputValues()

new_system.addOutputData(
    np.interp(new_time, modulated_time_output, modulated_output_signal)
)
stateSpace_modulated = StateSpace(systemInput = new_system.interpolatedInputValues[0],
                        systemOutput = new_system.outputValues, energyThreshold=1-1e-9)

A_modulated, B_modulated, C_modulated, D_modulated, initialState_modulated = stateSpace_modulated.buildStateSpaceSystem()


x_id_new, y_id_new = stateSpace_modulated.evolveInput(A=A_modulated, B=B_modulated, C=C_modulated, D=D_modulated, u=new_system.interpolatedInputValues[0], x0=initialState_modulated)

#%%

## Plotting initial input and reconstructed training output ##

plt.plot(new_system.timeOutput*1e9, new_system.interpolatedInputValues[0], label='Input: Current on Nodal Source')
plt.xlabel('Time (ns)')
plt.ylabel('Current (A)')
# plt.xlim((0, 2e-10))
plt.legend()
plt.grid()
plt.savefig("../../../../fig/Rogowski_modulated_input_data.png", dpi=300, bbox_inches='tight')
plt.show()


plt.plot(new_system.timeOutput*1e9, new_system.outputValues[0]*1e3, color='r', label='Output: Current on Coil')
# plt.plot(new_system.timeOutput*1e9, y_id_new[0]*1e3, '--', label='Output Reconstructed with SSSI')
plt.xlabel('Time (ns)')
plt.ylabel('Current (mA)')
plt.legend()
plt.grid()
plt.savefig("../../../../fig/Rogowski_modulated_output_data.png", dpi=300, bbox_inches='tight')
plt.show()
# %% Prediction with the previous parameters

modulated_finalTime = np.arange(0, 60e-9 + new_step, new_step)
modulated_finalOutput = np.interp(
    modulated_finalTime,
    modulated_time_output,
    modulated_output_signal
).reshape((1, -1))

# input
modulated_finalInput = np.interp(
    modulated_finalTime,
    modulated_time_input,
    modulated_input_signal
).reshape((1, -1))

x_id_new_predicted, y_id_new_predicted = stateSpace_modulated.evolveInput(A=A_modulated, B=B_modulated, C=C_modulated, D=D_modulated, u=modulated_finalInput[0], x0=initialState_modulated)

plt.plot(modulated_finalTime, modulated_finalOutput[0], label='Original Output on Coil')
plt.plot(modulated_finalTime, y_id_new_predicted[0], '--', label='SSSI method Output')
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
# %% Reconstruction of a larger Gaussian dataset with the previous parameters

data = np.load("large_gaussian_data.npz")

time_input = data["time_input"]
input_signal = data["input_signal"]
time_output_raw = data["time_output"]
output_signal_raw = data["output_signal"]

finalTime = np.arange(0, 45e-9 + new_step, new_step)
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

x_id_predicted, y_id_predicted = stateSpace_modulated.evolveInput(A=A_modulated, B=B_modulated, C=C_modulated, D=D_modulated, u=finalInput[0], x0=initialState_modulated)

plt.plot(finalTime, finalOutput[0], label='Original Output on Coil')
plt.plot(finalTime, y_id_predicted[0], '--', label='SSSI method Output')
plt.xlabel('Time')
plt.ylabel('Current')
plt.legend()
plt.grid()
plt.show()

# %%

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax = plt.subplots()


ax.plot(finalTime*1e9, finalOutput[0]*1e3,
        label='FDTD simulation Output', linewidth=3, color='k')

ax.plot(finalTime*1e9, y_id_predicted[0]*1e3,
        '--', label='Projection method:\nreconstruction from\ndifferent Output', color='r')

ax.set_xlabel('Time (ns)')
ax.set_ylabel('Current (mA)')
ax.legend(loc='upper left')
ax.grid()


axins = inset_axes(ax, width="45%", height="45%", loc='upper right')
axins.plot(finalTime*1e9, finalOutput[0]*1e3, color='k', linewidth=3)
axins.plot(finalTime*1e9, y_id_predicted[0]*1e3, '--^', color='r',
              markersize=8, markevery=20, markerfacecolor='none')
axins.set_xlim(32, 34)
mask = (finalTime >= 32e-9) & (finalTime <= 34e-9)
y_zoom = finalOutput[0][mask] * 1e3
axins.set_ylim(min(y_zoom) - 0.5, max(y_zoom) + 0.5)
axins.grid()

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.savefig("../../../../fig/Rogowski_modulated_output_reconstruction.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
