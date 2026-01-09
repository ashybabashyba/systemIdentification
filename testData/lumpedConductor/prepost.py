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
# Ramp excitation system identification

ramp_system = SystemIdentificationWrapper(timeInput=np.loadtxt("rampExcitation.exc", usecols=0),
                                          timeOutput=np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0))

ramp_system.addInputData(np.loadtxt("rampExcitation.exc", usecols=1))
ramp_system.buildInterpolatedInputValues()

plt.plot(ramp_system.timeInput, ramp_system.inputValues[0], label='Input Voltage')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.grid()
plt.legend()
plt.show()

for i in range(5):
    data = np.loadtxt(f"currentOutput{i}.dat", skiprows=1, usecols=1)
    ramp_system.addOutputData(data)


ramp_sys_id = system_identification(ramp_system.outputValues, ramp_system.interpolatedInputValues, "N4SID", SS_fixed_order=10)
xid_ramp, yid_ramp = fsetSIM.SS_lsim_process_form(ramp_sys_id.A, ramp_sys_id.B, 
                                        ramp_sys_id.C, ramp_sys_id.D, 
                                        ramp_system.interpolatedInputValues, ramp_sys_id.x0)

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten() 

for i in range(ramp_system.outputValues.shape[0]):
    axs[i].plot(ramp_system.timeOutput, ramp_system.outputValues[i, :], label='Original Output')
    axs[i].plot(ramp_system.timeOutput, yid_ramp[i, :], '--', label='N4SID')
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("Current Output")
    axs[i].set_title(f"Current Output {i} - SysID (Order 10)")
    axs[i].grid()
    axs[i].legend()

if ramp_system.outputValues.shape[0] < 6:
    axs[-1].axis('off')

plt.tight_layout()
plt.show()


# %% 
# Ramp excitation, prediction using the half of the total time

half_ramp_system = SystemIdentificationWrapper(timeInput=np.loadtxt("rampExcitation.exc", usecols=0),
                                               timeOutput=np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0))

half_ramp_system.addInputData(np.loadtxt("rampExcitation.exc", usecols=1))
half_ramp_system.buildInterpolatedInputValues()

for i in range(5):
    data = np.loadtxt(f"currentOutput{i}.dat", skiprows=1, usecols=1)
    half = len(data) // 2
    data = data[:half]
    half_ramp_system.addOutputData(data)

halfTimeOutput = half_ramp_system.timeOutput[:len(half_ramp_system.timeOutput)//2]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()  

for i in range(half_ramp_system.outputValues.shape[0]):  
    axes[i].plot(halfTimeOutput, half_ramp_system.outputValues[i])
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Current")
    axes[i].set_title(f"Current Output {i} - Half Output Data")
    axes[i].grid()

for j in range(half_ramp_system.outputValues.shape[0], 6):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


half_ramp_sys_id = system_identification(half_ramp_system.outputValues, 
                                         half_ramp_system.interpolatedInputValues, 
                                         "N4SID", SS_fixed_order=10)
xid_half_ramp, yid_half_ramp = fsetSIM.SS_lsim_process_form(half_ramp_sys_id.A, half_ramp_sys_id.B, 
                                                            half_ramp_sys_id.C, half_ramp_sys_id.D, 
                                                            half_ramp_system.interpolatedInputValues, 
                                                            half_ramp_sys_id.x0)

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()  

for i in range(half_ramp_system.outputValues.shape[0]):
    axes[i].plot(ramp_system.timeOutput, ramp_system.outputValues[i, :], label='Original Output')
    axes[i].plot(half_ramp_system.timeOutput, yid_half_ramp[i, :], '--', label='N4SID')
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Current")
    axes[i].set_title(f"Current Output {i} - SysID (Order 10)")
    axes[i].grid()
    axes[i].legend()

for j in range(half_ramp_system.outputValues.shape[0], 6):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
# %% 
# Gaussian excitation system identification

gaussian_system = SystemIdentificationWrapper(timeInput=np.loadtxt("gaussianExcitation.exc", usecols=0),
                                              timeOutput=np.loadtxt("gaussianOutput0.dat", skiprows=1, usecols=0))

gaussian_system.addInputData(np.loadtxt("gaussianExcitation.exc", usecols=1))
gaussian_system.buildInterpolatedInputValues()

for i in range(5):
    data = np.loadtxt(f"gaussianOutput{i}.dat", skiprows=1, usecols=1)
    gaussian_system.addOutputData(data)

plt.plot(gaussian_system.timeInput, gaussian_system.inputValues[0], label='Input Voltage')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.grid()
plt.legend()
plt.show()

gaussian_sys_id = system_identification(gaussian_system.outputValues, 
                                        gaussian_system.interpolatedInputValues, 
                                        "N4SID", IC="AIC")
xid_gaussian, yid_gaussian = fsetSIM.SS_lsim_process_form(gaussian_sys_id.A, gaussian_sys_id.B, 
                                                          gaussian_sys_id.C, gaussian_sys_id.D, 
                                                          gaussian_system.interpolatedInputValues, 
                                                          gaussian_sys_id.x0)

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()  

for i in range(gaussian_system.outputValues.shape[0]):
    axes[i].plot(gaussian_system.timeOutput, gaussian_system.outputValues[i, :], label='Original Output')
    axes[i].plot(gaussian_system.timeOutput, yid_gaussian[i, :], '--', label='N4SID')
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Current")
    axes[i].set_title(f"Current Output {i} - Information criteria AIC")
    axes[i].grid()
    axes[i].legend()

for j in range(gaussian_system.outputValues.shape[0], 6):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# %% 
# Prediction of Gaussian excitation based on parameters from ramp to excitation
# We only change the input values in this function respect to the ramp case
xid_prediction, yid_prediction = fsetSIM.SS_lsim_process_form(ramp_sys_id.A, ramp_sys_id.B, 
                                                             ramp_sys_id.C, ramp_sys_id.D, 
                                                             gaussian_system.interpolatedInputValues, 
                                                             ramp_sys_id.x0)

# We compare the results with the original outputs from the gaussian case
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()  

for i in range(gaussian_system.outputValues.shape[0]):
    axes[i].plot(gaussian_system.timeOutput, gaussian_system.outputValues[i, :], label='Original Output')
    axes[i].plot(gaussian_system.timeOutput, yid_prediction[i, :], '--', label='N4SID')
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Current")
    axes[i].set_title(f"Current Output {i} - Gaussian prediction")
    axes[i].grid()
    axes[i].legend()

for j in range(gaussian_system.outputValues.shape[0], 6):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
# %% 
# Ramp excitation, prediction using Data spaced every 10 points

eq_ramp_system = SystemIdentificationWrapper(timeInput=np.loadtxt("rampExcitation.exc", usecols=0),
                                               timeOutput=np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0)[::10])

eq_ramp_system.addInputData(np.loadtxt("rampExcitation.exc", usecols=1))
eq_ramp_system.buildInterpolatedInputValues()

for i in range(5):
    data = np.loadtxt(f"currentOutput{i}.dat", skiprows=1, usecols=1)
    eq_ramp_system.addOutputData(data[::10])

eqTimeOutput = eq_ramp_system.timeOutput

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()  

for i in range(eq_ramp_system.outputValues.shape[0]):  
    axes[i].plot(eqTimeOutput, eq_ramp_system.outputValues[i])
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Current")
    axes[i].set_title(f"Current Output {i} - Data spaced every 10 points")
    axes[i].grid()

for j in range(eq_ramp_system.outputValues.shape[0], 6):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


eq_ramp_sys_id = system_identification(eq_ramp_system.outputValues, 
                                         eq_ramp_system.interpolatedInputValues, 
                                         "N4SID", IC="AIC")
xid_eq_ramp, yid_eq_ramp = fsetSIM.SS_lsim_process_form(eq_ramp_sys_id.A, eq_ramp_sys_id.B, 
                                                            eq_ramp_sys_id.C, eq_ramp_sys_id.D, 
                                                            eq_ramp_system.interpolatedInputValues, 
                                                            eq_ramp_sys_id.x0)

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()  

for i in range(eq_ramp_system.outputValues.shape[0]):
    axes[i].plot(ramp_system.timeOutput, ramp_system.outputValues[i, :], label='Original Output')
    axes[i].plot(eq_ramp_system.timeOutput, yid_eq_ramp[i, :], '--', label='N4SID')
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Current")
    axes[i].set_title(f"currentOutput{i} - Information Criteria AIC")
    axes[i].grid()
    axes[i].legend()

for j in range(eq_ramp_system.outputValues.shape[0], 6):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# %% State Space identification using Naishadham(2016) and Juang(1997) method

step = 0.1e-9
initialTrainingTime = 0
finalTrainingTime = 10e-9
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

system = SystemIdentificationWrapper(timeInput=np.loadtxt("rampExcitation.exc", usecols=0),
                                     timeOutput=newTimeVector)

system.addInputData(np.loadtxt("rampExcitation.exc", usecols=1))
system.buildInterpolatedInputValues()
# Only works for one input and one output at the moment
system.addOutputData(np.interp(newTimeVector, 
                               np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0), 
                               np.loadtxt("currentOutput0.dat", skiprows=1, usecols=1)))



stateSpace = StateSpace(systemInput = system.interpolatedInputValues[0],
                        systemOutput = system.outputValues,
                        energyThreshold=1-1e-9)

A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()

H = stateSpace.buildHankelMatrix()

# Checking if the Hankel matrix is constructed properly
for i in range(H.shape[1] - 1):
    assert np.allclose(H[:-1, i+1], H[1:, i]), "Hankel matrix construction error!"

## Plotting reconstructed training output ##

# xid, yid = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=system.interpolatedInputValues[0], x0=initialState)
# plt.plot(system.timeOutput, system.outputValues[0], label='Original Output')
# plt.plot(system.timeOutput, yid[0], '--', label='Naishadham method Output')
# plt.xlabel('Time')
# plt.ylabel('Current')
# plt.ylim((-0.0002, 0.0012))
# plt.legend()
# plt.grid()
# plt.show()

## Prediction ##

finalTime = np.arange(0, 50e-9 + step, step)
finalOutput = np.interp(finalTime, 
                   np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0), 
                   np.loadtxt("currentOutput0.dat", skiprows=1, usecols=1)).reshape((1, -1))

finalInput = np.interp(finalTime, 
                   np.loadtxt("rampExcitation.exc", usecols=0), 
                   np.loadtxt("rampExcitation.exc", usecols=1)).reshape((1, -1))

x_id_predicted, y_id_predicted = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)

plt.plot(finalTime, finalOutput[0], label='Original Output')
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
step = 0.1e-9
newTimeVector = np.arange(0, 35e-9 + step, step)

systemMulti = SystemIdentificationWrapper(timeInput=np.loadtxt("rampExcitation.exc", usecols=0),
                                          timeOutput=newTimeVector)

systemMulti.addInputData(np.loadtxt("rampExcitation.exc", usecols=1)[::-1])
systemMulti.buildInterpolatedInputValues()

systemMulti.addOutputData(np.interp(newTimeVector, 
                               np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0), 
                               np.loadtxt("currentOutput0.dat", skiprows=1, usecols=1)[::-1]))

systemMulti.addOutputData(np.interp(newTimeVector, 
                               np.loadtxt("currentOutput1.dat", skiprows=1, usecols=0), 
                               np.loadtxt("currentOutput1.dat", skiprows=1, usecols=1)[::-1]))

stateSpaceMulti = StateSpace(systemInput = systemMulti.interpolatedInputValues[0],
                        systemOutput = systemMulti.outputValues,
                        energyThreshold=1-1e-6)

H = stateSpaceMulti.buildHankelMatrix()
for i in range(H.shape[1] - 1):
    assert np.allclose(H[:-stateSpaceMulti.numberOfOutputs, i+1], H[stateSpaceMulti.numberOfOutputs:, i]), "Hankel matrix construction error!"

A, B, C, D, initialState = stateSpaceMulti.buildStateSpaceSystem()
xid, yid = stateSpaceMulti.evolveInput(A=A, B=B, C=C, D=D, u=systemMulti.interpolatedInputValues[0], x0=initialState)

# %%
new_freqs = np.geomspace(1e6, 5e8, num=1000)

I_predicted = y_id_predicted.squeeze()
t_predicted = finalTime

I = finalOutput[0]
t = finalTime

I_f_predicted = np.array([np.sum(I_predicted * np.exp(-1j * 2 * np.pi * f * t_predicted)) for f in new_freqs])
I_f = np.array([np.sum(I * np.exp(-1j * 2 * np.pi * f * t)) for f in new_freqs])


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
