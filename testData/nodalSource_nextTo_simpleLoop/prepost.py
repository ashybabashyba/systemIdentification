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

from sippy_unipi import functionset as fset
from sippy_unipi import functionsetSIM as fsetSIM


# %% 
# System identification. Current nodal source next to simple loop

system = SystemIdentificationWrapper(timeInput=np.loadtxt("gaussianExcitation.exc", usecols=0),
                                     timeOutput=np.loadtxt("outputOnLoop.dat", skiprows=1, usecols=0))

system.addInputData(np.loadtxt("gaussianExcitation.exc", usecols=1))
system.buildInterpolatedInputValues()

system.addOutputData(np.loadtxt("outputOnLoop.dat", skiprows=1, usecols=1))
system.addOutputData(np.loadtxt("outputOnNodalSource.dat", skiprows=1, usecols=1))


plt.plot(system.timeInput, system.inputValues[0], label='Input Current')
plt.plot(system.timeOutput, system.outputValues[1, :], '--', label='Output Current on Nodal Line')
plt.xlabel('Time')
plt.ylabel('Current')
plt.legend()
plt.grid()
plt.show()

plt.plot(system.timeOutput, system.outputValues[0, :], label='Output Current on Loop')
plt.xlabel('Time')
plt.ylabel('Current')
plt.legend()
plt.grid()
plt.show()

sys_id = system_identification(system.outputValues, system.interpolatedInputValues, "N4SID", IC="AIC")
xid, yid = fsetSIM.SS_lsim_process_form(sys_id.A, sys_id.B, 
                                        sys_id.C, sys_id.D, 
                                        system.interpolatedInputValues, sys_id.x0)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs = axs.flatten() 

for i in range(system.outputValues.shape[0]):
    axs[i].plot(system.timeOutput, system.outputValues[i, :], label='Original Output')
    axs[i].plot(system.timeOutput, yid[i, :], '--', label='N4SID')
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("Current Output")
    if i==0:
        axs[i].set_title("Current On Loop - Information Criteria AIC")
    else:
        axs[i].set_title("Current on Nodal Source - Information Criteria AIC")
    axs[i].grid()
    axs[i].legend()


plt.tight_layout()
plt.show()
# %%
