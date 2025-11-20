# %%
import numpy as np
import matplotlib.pyplot as plt

try:
    from sippy_unipi import system_identification
except ImportError:
    import os
    import sys

    sys.path.append(os.pardir)
    from sippy_unipi import system_identification

from sippy_unipi import functionset as fset
from sippy_unipi import functionsetSIM as fsetSIM


# %% Load Data

voltageInput = np.loadtxt("rampExcitation.exc", usecols=1)
timeInput    = np.loadtxt("rampExcitation.exc", usecols=0)

plt.plot(timeInput, voltageInput, label='Input Current')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.grid()
plt.show()


# %% Trying to reproduce the output using SIPPY

currentOutput = np.loadtxt("currentOutput0.dat", skiprows=1, usecols=1)
timeOutput    = np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0)

# First interpolate
newVoltageInput = np.interp(timeOutput, timeInput, voltageInput)

U = newVoltageInput.reshape(1, -1)
y = currentOutput.reshape(1, -1)


plt.plot(timeOutput, y[0])
plt.ylabel("Current")
plt.grid()
plt.xlabel("Time")
plt.title("Current at the voltage source - System Identification (Order 10)")

##System identification
METHOD = ["N4SID", "CVA", "MOESP"]
lege = ["Original Output"]
for i in range(len(METHOD)):
    method = METHOD[i]
    sys_id = system_identification(y, U, method, SS_fixed_order=10)
    xid, yid = fsetSIM.SS_lsim_process_form(
        sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0
    )
    #
    plt.plot(timeOutput, yid[0])
    lege.append(method)
plt.legend(lege)
plt.show()

# %%

currentOutput = np.loadtxt("currentOutput1.dat", skiprows=1, usecols=1)
timeOutput    = np.loadtxt("currentOutput1.dat", skiprows=1, usecols=0)

# First interpolate
newVoltageInput = np.interp(timeOutput, timeInput, voltageInput)

U = newVoltageInput.reshape(1, -1)
y = currentOutput.reshape(1, -1)


plt.plot(timeOutput, y[0])
plt.ylabel("Current")
plt.grid()
plt.xlabel("Time")
plt.title("Current before lumped cell - System Identification (Order 10)")

##System identification
METHOD = ["N4SID"]
lege = ["Original Output"]
for i in range(len(METHOD)):
    method = METHOD[i]
    sys_id = system_identification(y, U, method, SS_fixed_order=10)
    xid, yid = fsetSIM.SS_lsim_process_form(
        sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0
    )
    #
    plt.plot(timeOutput, yid[0])
    lege.append(method)
plt.legend(lege)
plt.show()
# %%

currentOutput = np.loadtxt("currentOutput2.dat", skiprows=1, usecols=1)
timeOutput    = np.loadtxt("currentOutput2.dat", skiprows=1, usecols=0)

# First interpolate
newVoltageInput = np.interp(timeOutput, timeInput, voltageInput)

U = newVoltageInput.reshape(1, -1)
y = currentOutput.reshape(1, -1)


plt.plot(timeOutput, y[0])
plt.ylabel("Current")
plt.grid()
plt.xlabel("Time")
plt.title("Current at the start of the lumped cell - System Identification (Order 10)")

##System identification
METHOD = ["N4SID", "CVA", "MOESP"]
lege = ["Original Output"]
for i in range(len(METHOD)):
    method = METHOD[i]
    sys_id = system_identification(y, U, method, SS_fixed_order=10)
    xid, yid = fsetSIM.SS_lsim_process_form(
        sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0
    )
    #
    plt.plot(timeOutput, yid[0])
    lege.append(method)
plt.legend(lege)
plt.show()

# %%

currentOutput = np.loadtxt("currentOutput3.dat", skiprows=1, usecols=1)
timeOutput    = np.loadtxt("currentOutput3.dat", skiprows=1, usecols=0)

# First interpolate
newVoltageInput = np.interp(timeOutput, timeInput, voltageInput)

U = newVoltageInput.reshape(1, -1)
y = currentOutput.reshape(1, -1)


plt.plot(timeOutput, y[0])
plt.ylabel("Current")
plt.grid()
plt.xlabel("Time")
plt.title("Current at the end of the lumped cell - System Identification (Order 10)")

##System identification
METHOD = ["N4SID", "CVA", "MOESP"]
lege = ["Original Output"]
for i in range(len(METHOD)):
    method = METHOD[i]
    sys_id = system_identification(y, U, method, SS_fixed_order=10)
    xid, yid = fsetSIM.SS_lsim_process_form(
        sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0
    )
    #
    plt.plot(timeOutput, yid[0])
    lege.append(method)
plt.legend(lege)
plt.show()

# %%

currentOutput = np.loadtxt("currentOutput4.dat", skiprows=1, usecols=1)
timeOutput    = np.loadtxt("currentOutput4.dat", skiprows=1, usecols=0)

# First interpolate
newVoltageInput = np.interp(timeOutput, timeInput, voltageInput)

U = newVoltageInput.reshape(1, -1)
y = currentOutput.reshape(1, -1)


plt.plot(timeOutput, y[0])
plt.ylabel("Current")
plt.grid()
plt.xlabel("Time")
plt.title("Current after lumped cell - System Identification (Order 10)")

##System identification
METHOD = ["N4SID"]
lege = ["Original Output"]
for i in range(len(METHOD)):
    method = METHOD[i]
    sys_id = system_identification(y, U, method, SS_fixed_order=10)
    xid, yid = fsetSIM.SS_lsim_process_form(
        sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0
    )
    #
    plt.plot(timeOutput, yid[0])
    lege.append(method)
plt.legend(lege)
plt.show()
# %% Lets try SSSI with single input and multiple outputs

currentOutput0 = np.loadtxt("currentOutput0.dat", skiprows=1, usecols=1)
currentOutput1 = np.loadtxt("currentOutput1.dat", skiprows=1, usecols=1)
currentOutput2 = np.loadtxt("currentOutput2.dat", skiprows=1, usecols=1)
currentOutput3 = np.loadtxt("currentOutput3.dat", skiprows=1, usecols=1)
currentOutput4 = np.loadtxt("currentOutput4.dat", skiprows=1, usecols=1)

assert np.all(np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0) == np.loadtxt("currentOutput1.dat", skiprows=1, usecols=0))
assert np.all(np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0) == np.loadtxt("currentOutput2.dat", skiprows=1, usecols=0))
assert np.all(np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0) == np.loadtxt("currentOutput3.dat", skiprows=1, usecols=0))
assert np.all(np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0) == np.loadtxt("currentOutput4.dat", skiprows=1, usecols=0))

timeOutput    = np.loadtxt("currentOutput0.dat", skiprows=1, usecols=0)

newVoltageInput = np.interp(timeOutput, timeInput, voltageInput)

U = newVoltageInput.reshape(1, -1)
y = np.vstack((currentOutput0, currentOutput1, currentOutput2, currentOutput3, currentOutput4))


for i in range(y.shape[0]):
    METHOD = ["N4SID"]
    lege = ["Original Output"]
    
    plt.plot(timeOutput, y[i, :])
    plt.ylabel("Current Output {}".format(i))
    plt.grid()
    plt.xlabel("Time")
    plt.title("Current Output {} - System Identification (Order 10)".format(i))

    for i in range(len(METHOD)):
        method = METHOD[i]
        sys_id = system_identification(y, U, method, SS_fixed_order=10)
        xid, yid = fsetSIM.SS_lsim_process_form(
                sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0
            )

        plt.plot(timeOutput, yid[i, :])
        lege.append(method)
    plt.legend(lege)
    plt.show()

# assert np.all(yid[0,:] != yid[1, :])
# assert np.all(yid[0,:] != yid[2, :])
# assert np.all(yid[0,:] != yid[3, :])
# assert np.all(yid[0,:] != yid[4, :])

# assert np.all(yid[1, :] != yid[2, :])
# assert np.all(yid[1, :] != yid[3, :])
# assert np.all(yid[1, :] != yid[4, :])

# assert np.all(yid[2, :] == yid[3, :])
# assert np.all(yid[2, :] != yid[4, :])

# %%
