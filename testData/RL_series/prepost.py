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

# %% R=50, L=1e-3
step = 0.01e-3
initialTrainingTime = 0
finalTrainingTime = 1.75e-3
newTimeVector = np.arange(initialTrainingTime, finalTrainingTime + step, step)

dataFile = "dataSet_seriesRL"


system = SystemIdentificationWrapper(timeInput=np.loadtxt(dataFile, usecols=0, skiprows=1),
                                     timeOutput=newTimeVector)

system.addInputData(np.loadtxt(dataFile, usecols=3, skiprows=1))
system.buildInterpolatedInputValues()
# Voltage at R
system.addOutputData(np.interp(newTimeVector, 
                               np.loadtxt(dataFile, skiprows=1, usecols=0), 
                               np.loadtxt(dataFile, skiprows=1, usecols=1)))
# # Voltage at L
# system.addOutputData(np.interp(newTimeVector, 
#                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
#                                np.loadtxt(dataFile, skiprows=1, usecols=2)))
# # Voltage at source
# system.addOutputData(np.interp(newTimeVector, 
#                                np.loadtxt(dataFile, skiprows=1, usecols=0), 
#                                np.loadtxt(dataFile, skiprows=1, usecols=3)))
# Current
system.addOutputData(np.interp(newTimeVector, 
                               np.loadtxt(dataFile, skiprows=1, usecols=0), 
                               np.loadtxt(dataFile, skiprows=1, usecols=4)))

stateSpace = StateSpace(systemInput = system.interpolatedInputValues[0],
                        systemOutput = system.outputValues)

A, B, C, D, initialState = stateSpace.buildStateSpaceSystem()
#%%
from scipy import signal

# sistema discreto
sys = signal.dlti(A, B, C, D, dt=step)

# convertir a función de transferencia
num, den = signal.ss2tf(A, B, C, D)

# polos discretos
z_poles = np.roots(den)

# convertir a continuo
poles_cont = np.log(z_poles) / step

# eigvals = np.linalg.eigvals(A)
# poles_cont = np.log(eigvals) / step

# quedarse con polos casi reales
real_poles = poles_cont[np.abs(np.imag(poles_cont)) < 1e-6]

# quedarse con los negativos
real_poles = real_poles[np.real(real_poles) < 0]
p_phys = real_poles[np.argmax(np.real(real_poles))]

# %%

finalTime = np.arange(0, 5e-3 + step, step)
finalOutput = np.interp(finalTime, 
                   np.loadtxt(dataFile, skiprows=1, usecols=0), 
                   np.loadtxt(dataFile, skiprows=1, usecols=3)).reshape((1, -1))

finalInput = np.interp(finalTime, 
                   np.loadtxt(dataFile, usecols=0, skiprows=1), 
                   np.loadtxt(dataFile, usecols=4, skiprows=1)).reshape((1, -1))

_, y_id = stateSpace.evolveInput(A=A, B=B, C=C, D=D, u=finalInput[0], x0=initialState)


# plt.plot(finalTime, finalOutput[0], label='Voltage at source')
plt.plot(finalTime, y_id[1], '--', label='Reconstructed voltage')
plt.xlabel('Time')
plt.ylabel('Voltage')
# plt.xlim(0, 1e-3)
plt.legend()
plt.grid()
plt.show()

# %% DTFT for impedance and cutoff frequency estimation

new_freqs = np.geomspace(1e2, 5e3, num=1000)

# I =   y_id[3].squeeze()
I = finalInput[0]
V_L = y_id[1].squeeze()
t = finalTime


I_f = np.array([np.sum(I * np.exp(-1j * 2 * np.pi * f * t)) for f in new_freqs])
V_L_f = np.array([np.sum(V_L * np.exp(-1j * 2 * np.pi * f * t)) for f in new_freqs])


mask_I_f_neq_0 = I_f != 0
Z_L = V_L_f[mask_I_f_neq_0] / I_f[mask_I_f_neq_0]



plt.figure()
plt.plot(new_freqs, np.abs(Z_L), '.', label='Impedance of inductor', color='red')
plt.plot(new_freqs, 2*np.pi*new_freqs*1e-3, '--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Impedance')
plt.grid(which='both')
plt.legend()
plt.show()

# %%
