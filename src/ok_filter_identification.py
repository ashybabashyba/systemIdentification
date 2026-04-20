import numpy as np
from scipy.signal import dlsim

class ObserverKalmanFilterIdentification():
    def __init__(self, A, B, C, D, initialState, energyThreshold=0.99):
        self.A_est = A
        self.B_est = B
        self.C_est = C
        self.D_est = D
        self.initialState = initialState

        self.energyThreshold = energyThreshold

        self.inputValues             = []

        self.referenceOutputValues    = []
        self.interpolatedReferenceOutputValues = []

        self.deterministicOutputValues = []
        self.interpolatedDeterministicOutputValues = []

        self.inputTimeData  = []
        self.deterministicOutputTimeData = []
        self.referenceOutputTimeData = []

    def addInputData(self, inputData, inputTimeData):
        if len(self.inputValues) == 0:
            self.inputValues = np.array([inputData])
        else:
            self.inputValues = np.vstack((self.inputValues, inputData))
        

        if len(self.inputTimeData) != 0:
            if not np.array_equal(self.inputTimeData, inputTimeData):
                raise ValueError("Input time data must be consistent across calls")
        else:
            self.inputTimeData = inputTimeData

    def addDeterministicOutput(self, outputData, outputTimeData):
        if len(self.deterministicOutputValues) == 0:
            self.deterministicOutputValues = np.array([outputData])
        else:
            self.deterministicOutputValues = np.vstack((self.deterministicOutputValues, outputData))

        
        if len(self.deterministicOutputTimeData) != 0:
            if not np.array_equal(self.deterministicOutputTimeData, outputTimeData):
                raise ValueError("Output time data must be consistent across calls")
        else:
            self.deterministicOutputTimeData = outputTimeData

    def addReferenceOutput(self, outputData, outputTimeData):
        if len(self.referenceOutputValues) == 0:
            self.referenceOutputValues = np.array([outputData])
        else:
            self.referenceOutputValues = np.vstack((self.referenceOutputValues, outputData))

        
        if len(self.referenceOutputTimeData) != 0:
            if not np.array_equal(self.referenceOutputTimeData, outputTimeData):
                raise ValueError("Output time data must be consistent across calls")
        else:
            self.referenceOutputTimeData = outputTimeData

    def buildInterpolatedOutputs(self):
        t_ref_max = self.referenceOutputTimeData[-1]
        k_prime = np.searchsorted(self.inputTimeData, t_ref_max, side='right') - 1
        self.k_prime = k_prime

        t_common = self.inputTimeData[:k_prime + 1]

        y_ref = np.array(self.referenceOutputValues)
        y_det = np.array(self.deterministicOutputValues)

        if y_ref.ndim == 1:
            y_ref = y_ref.reshape(-1, 1)

        y_ref_interp = []
        y_det_interp = []

        for i in range(y_ref.shape[0]):
            y_ref_interp.append(
                np.interp(
                    t_common,
                    self.referenceOutputTimeData,
                    y_ref[i, :]
                )
            )

        for i in range(y_det.shape[0]):
            y_det_interp.append(
                np.interp(
                    t_common,
                    self.deterministicOutputTimeData,
                    y_det[i, :]
                )
            )

        self.interpolatedReferenceOutputValues = np.vstack(y_ref_interp)
        self.interpolatedDeterministicOutputValues = np.vstack(y_det_interp)

    def buildOutputResiduals(self):
        self.buildInterpolatedOutputs()

        return self.interpolatedReferenceOutputValues - self.interpolatedDeterministicOutputValues
    
    def buildCorrelationVector(self, residuals):
        n = len(residuals)

        residuals_norm = residuals - np.mean(residuals)

        rho = np.correlate(residuals_norm, residuals_norm, mode='full') 
        rho_raw = rho[n-1:]
        rho_normalized = rho_raw / rho_raw[0]

        return rho_normalized

    def energyCriterionForTruncation(self, residuals, order=1):
        autocorrelationVector = self.buildCorrelationVector(residuals)
        rho_abs = np.abs(autocorrelationVector)

        total_energy = np.sum(rho_abs**order)
        cumulative_energy = np.cumsum(rho_abs**order) / total_energy

        p = np.where(cumulative_energy >= self.energyThreshold)[0][0]

        return p
    
    def buildRegressionMatrix(self, residuals, p):
        if residuals.ndim == 1:
            residuals = residuals.reshape(1, -1)
            
        n_outputs, L = residuals.shape
        V2 = np.zeros((n_outputs*p, L))

        for i in range(p):
            row_start = i * n_outputs
            row_end = (i + 1) * n_outputs
            
            if i == 0:
                V2[row_start:row_end, :] = residuals
            else:
                V2[row_start:row_end, i:] = residuals[:, :L-i]


        return V2

    def buildFilterObservability(self):
        residuals = self.buildOutputResiduals()
        p = self.energyCriterionForTruncation(residuals)

        if residuals.ndim == 1:
            residuals = residuals.reshape(1, -1)
        n_outputs, L = residuals.shape

        V2 = self.buildRegressionMatrix(residuals, p)
        y_2 = np.asarray(residuals)

        Y_2_bar = -y_2 @ np.linalg.pinv(V2, rcond=1e-24)


        Yo_list = []
        Y_bar_blocks = [Y_2_bar[:, i*n_outputs : (i+1)*n_outputs] for i in range(p)]

        for k in range(p):
            current_Yo = Y_bar_blocks[k].copy()
            
            for i in range(k):
                current_Yo += Yo_list[(k-1)-i] @ Y_bar_blocks[i]
                
            Yo_list.append(current_Yo)

        filterObservability = np.vstack(Yo_list)
        
        return filterObservability
    
    def buildObservabilityMatrix(self):
        residuals = self.buildOutputResiduals()
        p = self.energyCriterionForTruncation(residuals)

        n_outputs = self.C_est.shape[0]
        r = self.A_est.shape[0]

        Omega_rows = np.zeros((n_outputs * p, r))
        
        current_CA = self.C_est.copy()
        for k in range(p):
            Omega_rows[k*n_outputs : (k+1)*n_outputs, :] = current_CA
            current_CA = current_CA @ self.A_est.copy()  

        return Omega_rows
    
    def buildObserverGain(self):
        filterObservability = self.buildFilterObservability()
        observabilityMatrix = self.buildObservabilityMatrix()

        observerGain = np.linalg.lstsq(observabilityMatrix, filterObservability, rcond=None)[0]

        return observerGain