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
    
    def buildCorrelationVector(self):
        residuals = np.atleast_2d(self.residuals) 
        n_outputs, L = residuals.shape

        residuals_norm = residuals - np.mean(residuals, axis=1, keepdims=True)

        rho_total = np.zeros(L)

        for i in range(n_outputs):
            channel = residuals_norm[i, :]
            rho_channel = np.correlate(channel, channel, mode='full')
            
            rho_total += rho_channel[L-1:]

        rho_normalized = rho_total / rho_total[0]

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

        residuals = residuals[:, :-1]
            
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
        # p = self.energyCriterionForTruncation(residuals)

        if residuals.ndim == 1:
            residuals = residuals.reshape(1, -1)
        n_outputs, L = residuals.shape
        p = L - 1

        V2 = self.buildRegressionMatrix(p)
        y_2 = np.asarray(residuals)
        y_2 = y_2[:, 1:]

        U, S, Vt = np.linalg.svd(V2, full_matrices=False)

        tol = S[0] * self.energyThreshold
    
        S_inv = np.zeros_like(S)
        S_inv[S > tol] = 1.0 / S[S > tol]

        Y_2_bar = -y_2 @ (Vt.T @ np.diag(S_inv) @ U.T)
        # Y_2_bar = -y_2 @ np.linalg.pinv(V2, rcond=1e-24)

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
        # p = self.energyCriterionForTruncation(residuals)

        if residuals.ndim == 1:
            residuals = residuals.reshape(1, -1)
        n_outputs, L = residuals.shape
        p = L - 1

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
    
    def evolveWithFilter(self):
        G = self.buildObserverGain()
        A_cl = self.A_est - G @ self.C_est 

        u = np.array(self.inputValues)
        if u.ndim == 1:
            u = u.reshape(-1, 1)

        N = u.shape[1]
        n = self.A_est.shape[0]
        p = self.C_est.shape[0]

        x = np.zeros((n, N))
        y = np.zeros((p, N))

        Sigma_y = np.zeros((p, p, N))   
        sigma_y = np.zeros((p, N))      

        x[:, 0] = self.initialState
        y[:, 0] = self.C_est @ x[:, 0] + self.D_est @ u[:,0]

        if hasattr(self, 'residuals') and self.residuals.shape[1] > 1:
            Sigma_residual = np.cov(self.residuals[:, :self.k_prime])
            if Sigma_residual.ndim == 0:
                Sigma_residual = np.array([[Sigma_residual]])
        else:
            error_base = np.var(self.y_ref_interp[:, :self.k_prime] - y[:, :1])
            Sigma_residual = error_base * np.eye(p)

        P_estado_train = G @ Sigma_residual @ G.T
        P = P_estado_train.copy()
        
        Sigma_y[:, :, 0] = self.C_est @ P @ self.C_est.T + Sigma_residual
        sigma_y[:, 0] = np.sqrt(np.diag(Sigma_y[:, :, 0]))

        for k in range(1, N):
            x_pred = self.A_est @ x[:, k-1] + self.B_est @ u[:,k-1]
            y_pred = self.C_est @ x_pred + self.D_est @ u[:,k]

            Q_subspace = G @ Sigma_residual @ G.T

            if k <= self.k_prime:
                y_ref = self.y_ref_interp[:, k]
                if y_ref.ndim == 0:
                    y_ref = np.array([y_ref])

                innovation = y_ref - y_pred
                x[:, k] = x_pred - G @ innovation
                y[:, k] = self.C_est @ x[:, k] + self.D_est @ u[:,k] + innovation

                P_pred = self.A_est @ P @ self.A_est.T + Q_subspace
                I_GC = np.eye(n) + G @ self.C_est
                P = I_GC @ P_pred @ I_GC.T + G @ Sigma_residual @ G.T
                
                Sigma_y[:, :, k] = Sigma_residual

            else:
                x[:, k] = x_pred
                y[:, k] = y_pred
                
                P = self.A_est @ P @ self.A_est.T + Q_subspace
                
                Sigma_y[:, :, k] = self.C_est @ P @ self.C_est.T + Sigma_residual

            diags = np.diag(Sigma_y[:, :, k]).copy()
            diags[diags < 0] = 0.0
            sigma_y[:, k] = np.sqrt(diags)

        self.outputCovariance = Sigma_y
        self.outputStd = sigma_y

        return x, y