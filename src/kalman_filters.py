import numpy as np
from scipy.signal import dlsim

class KalmanProcessing():
    def __init__(self, A, B, C, D, initialState):
        self.A_est = A
        self.B_est = B
        self.C_est = C
        self.D_est = D
        self.initialState = initialState

        self.inputValues             = []
        self.referenceOutputValues    = []
        self.interpolatedReferenceOutputValues = []

        self.inputTimeData  = []
        self.outputTimeData = []

    def setKalmanProcessNoise(self):
        Q = np.zeros((self.A_est.shape[0], self.A_est.shape[1]))

        for i in range(self.A_est.shape[0]):
            Q[i, i] = 1e-7          # Cambiar por gaussiana, eventualmente converge pero igual

        self.Q = Q
        return self.Q

    def setKalmanMeasurementNoise(self):
        R = np.zeros((self.C_est.shape[0], self.C_est.shape[0]))

        for i in range(self.C_est.shape[0]):
            R[i, i] = 1e-7          # Cambiar por gaussiana, eventualmente converge pero igual

        self.R = R
        return self.R

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

    def addReferenceOutput(self, outputData, outputTimeData):
        if len(self.referenceOutputValues) == 0:
            self.referenceOutputValues = np.array([outputData])
        else:
            self.referenceOutputValues = np.vstack((self.referenceOutputValues, outputData))

        
        if len(self.outputTimeData) != 0:
            if not np.array_equal(self.outputTimeData, outputTimeData):
                raise ValueError("Output time data must be consistent across calls")
        else:
            self.outputTimeData = outputTimeData

    def buildInterpolatedReference(self):
        t_ref_max = self.outputTimeData[-1]
        k_prime = np.searchsorted(self.inputTimeData, t_ref_max, side='right') - 1
        self.k_prime = k_prime

        t_common = self.inputTimeData[:k_prime + 1]

        y_ref = np.array(self.referenceOutputValues)
        if y_ref.ndim == 1:
            y_ref = y_ref.reshape(-1, 1)

        y_interp = []
        for i in range(y_ref.shape[0]):
            y_interp.append(
                np.interp(
                    t_common,
                    self.outputTimeData,
                    y_ref[i, :]
                )
            )

        self.interpolatedReferenceOutputValues = np.vstack(y_interp)


    def evolveWithKalmanFilter(self):
        self.buildInterpolatedReference()

        u = np.array(self.inputValues)
        if u.ndim == 1:
            u = u.reshape(-1, 1)   # (N, m)

        N = u.shape[1]
        n = self.A_est.shape[0]
        p = self.C_est.shape[0]

        x = np.zeros((n, N))
        y = np.zeros((p, N))

        Sigma_y = np.zeros((p, p, N))   # covarianza completa
        sigma_y = np.zeros((p, N))      # desviación estándar 

        x[:, 0] = self.initialState
        y[:, 0] = self.C_est @ x[:, 0] + self.D_est @ u[:,0]

        P = np.eye(n)  # Se inicializa como identidad?
        Q = self.Q
        R = self.R

        Sigma_y[:, :, 0] = self.C_est @ P @ self.C_est.T
        sigma_y[:, 0] = np.sqrt(np.diag(Sigma_y[:, :, 0]))

        for k in range(1, N):
            x_pred = self.A_est @ x[:, k-1] + self.B_est @ u[:,k-1]
            P_pred = self.A_est @ P @ self.A_est.T + Q

            y_pred = self.C_est @ x_pred + self.D_est @ u[:,k]

            # Correction step
            if k <= self.k_prime:
                y_ref = self.interpolatedReferenceOutputValues[:, k]

                if y_ref.ndim == 0:
                    y_ref = np.array([y_ref])

                S = self.C_est @ P_pred @ self.C_est.T + R
                K = P_pred @ self.C_est.T @ np.linalg.inv(S)

                x[:, k] = x_pred + K @ (y_ref - y_pred)
                P = (np.eye(n) - K @ self.C_est) @ P_pred   # Inestable numericamente

            # No correction
            else:
                x[:, k] = x_pred
                P = P_pred

            y[:, k] = self.C_est @ x[:, k] + self.D_est @ u[:,k]

            Sigma_y[:, :, k] = self.C_est @ P @ self.C_est.T
            sigma_y[:, k] = np.sqrt(np.diag(Sigma_y[:, :, k]))

        self.stateTrajectory = x
        self.outputTrajectory = y

        self.outputCovariance = Sigma_y
        self.outputStd = sigma_y

        return x, y