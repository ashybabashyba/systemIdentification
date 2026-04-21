import numpy as np
from scipy.signal import dlsim

class ObserverKalmanFilterIdentification:
    def __init__(self, A, B, C, D, initialState, dataWrapper, energyThreshold=0.99):
        self.A_est = A
        self.B_est = B
        self.C_est = C
        self.D_est = D
        self.initialState = initialState

        self.energyThreshold = energyThreshold

        self.residuals = dataWrapper.buildOutputResiduals() 
        self.k_prime = dataWrapper.k_prime
        self.inputValues = dataWrapper.inputValues
        self.y_ref_interp = dataWrapper.interpolatedReferenceOutputValues

        self.G = None
        self.Q = None
        self.R = None
    
    def buildCorrelationVector(self):
        residuals = self.residuals
        n = len(residuals)

        residuals_norm = residuals - np.mean(residuals)

        rho = np.correlate(residuals_norm, residuals_norm, mode='full') # Esto puede tener problemas para multiple outputs
        rho_raw = rho[n-1:]
        rho_normalized = rho_raw / rho_raw[0]

        return rho_normalized

    def energyCriterionForTruncation(self, order=1):
        autocorrelationVector = self.buildCorrelationVector()
        rho_abs = np.abs(autocorrelationVector)

        total_energy = np.sum(rho_abs**order)
        cumulative_energy = np.cumsum(rho_abs**order) / total_energy

        p = np.where(cumulative_energy >= self.energyThreshold)[0][0]

        return p
    
    def buildRegressionMatrix(self, p):
        residuals = self.residuals
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
        residuals = self.residuals
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
        residuals = self.residuals
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
    
    def estimateNoiseCovariances(self, observerGain):
        y2 = self.residuals
        G = observerGain     

        n_states = self.A_est.shape[0]
        n_outputs, L = y2.shape
        
        x2_hat = np.zeros((n_states, 1))
        innovations = []

        A_obs = self.A_est + G @ self.C_est
        
        for k in range(L):
            y2_k = y2[:, k:k+1]
            
            eps_k = y2_k - self.C_est @ x2_hat
            innovations.append(eps_k)
            
            x2_hat = A_obs @ x2_hat - G @ y2_k

        eps_matrix = np.hstack(innovations)
        
        R = np.cov(eps_matrix, bias=True)
        Q = G @ R @ G.T
        
        return Q, R
    
    def runIdentification(self):
        self.G = self.buildObserverGain()
        self.Q, self.R = self.estimateNoiseCovariances(self.G)
    
    def evolveWithFilter(self):
        if self.G is None:
            self.runIdentification()

        u = np.array(self.inputValues)
        if u.ndim == 1:
            u = u.reshape(-1, 1)

        N = u.shape[1]
        n = self.A_est.shape[0]
        p = self.C_est.shape[0]

        x = np.zeros((n, N))
        y = np.zeros((p, N))

        Sigma_y = np.zeros((p, p, N))   # covarianza completa
        sigma_y = np.zeros((p, N))      # desviación estándar 

        x[:, 0] = self.initialState
        y[:, 0] = self.C_est @ x[:, 0] + self.D_est @ u[:,0]

        P = np.zeros((n, n))  # El estado inicial está inicializado en cero, por lo que no debería haber incertidumbre inicial

        Sigma_y[:, :, 0] = self.C_est @ P @ self.C_est.T
        sigma_y[:, 0] = np.sqrt(np.diag(Sigma_y[:, :, 0]))

        for k in range(1, N):
            x_pred = self.A_est @ x[:, k-1] + self.B_est @ u[:,k-1]
            P_pred = self.A_est @ P @ self.A_est.T + Q

            y_pred = self.C_est @ x_pred + self.D_est @ u[:,k]

            # Correction step
            if k <= self.k_prime:
                y_ref = self.y_ref_interp[:, k]

                if y_ref.ndim == 0:
                    y_ref = np.array([y_ref])

                S = self.C_est @ P_pred @ self.C_est.T + R
                K = np.linalg.solve(S.T, (P_pred @ self.C_est.T).T).T

                x[:, k] = x_pred + K @ (y_ref - y_pred)
                # P = (np.eye(n) - K @ self.C_est) @ P_pred   # Inestable numericamente
                P = (np.eye(n) - K @ self.C_est) @ P_pred @ (np.eye(n) - K @ self.C_est).T + K @ R @ K.T

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