import numpy as np
from scipy.signal import dlsim
from numba import njit

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
        self.k_prime_ctrl = dataWrapper.k_prime_ctrl
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
    
    def evolveWithFilter(self, x_ref):
        G = self.buildObserverGain()

        u = np.array(self.inputValues)
        if u.ndim == 1:
            u = u.reshape(-1, 1)

        N = u.shape[1]
        n = self.A_est.shape[0]
        p = self.C_est.shape[0]

        x = np.zeros((n, N))
        y = np.zeros((p, N))   

        epsilon = np.zeros((p, self.k_prime_ctrl + 1)) 

        x[:, 0] = self.initialState
        y[:, 0] = self.C_est @ x[:, 0] + self.D_est @ u[:,0]

        y_ref_0 = self.y_ref_interp[:, 0]
        if y_ref_0.ndim == 0: y_ref_0 = np.array([y_ref_0])
        epsilon[:, 0] = y_ref_0 - y[:, 0] - self.C_est @ (x[:, 0]- x_ref[:, 0])


        for k in range(1, N):
            x_pred = self.A_est @ x[:, k-1] + self.B_est @ u[:,k-1]
            y_pred = self.C_est @ x_pred + self.D_est @ u[:,k]

            if k <= self.k_prime_ctrl:
                y_ref = self.y_ref_interp[:, k]
                if y_ref.ndim == 0:
                    y_ref = np.array([y_ref])

                residual = y_ref - y_pred
                x[:, k] = x_pred - G @ epsilon[:, k-1]
                epsilon[:, k] = residual - self.C_est @ (x[:, k] - x_ref[:, k])
                y[:, k] = self.C_est @ x[:, k] + self.D_est @ u[:,k] + epsilon[:, k]

            else:
                x[:, k] = x_pred
                y[:, k] = y_pred

        self.epsilon = epsilon              

        return x, y
    
    def buildErrorCovariance(self, x_mean, x_1):
        E_escape = x_mean[:, :self.k_prime_ctrl + 1] - x_1[:, :self.k_prime_ctrl + 1]
        
        N_escape = E_escape.shape[1]
        P = (E_escape @ E_escape.T) / N_escape
        
        return P
    
    def computeUncertaintyFromEpsilon(self, P_escape):
        epsilon_est = self.epsilon
        N_pts = epsilon_est.shape[1]
        R = (epsilon_est @ epsilon_est.T) / N_pts
        if R.ndim == 0:
            R = np.array([[R]])

        G = self.buildObserverGain()
        Q = G @ R @ G.T

        u = np.array(self.inputValues)
        if u.ndim == 1:
            u = u.reshape(-1, 1)

        N = u.shape[1]
        n = self.A_est.shape[0]
        p = self.C_est.shape[0]      

        Sigma_y = np.zeros((p, p, N))   
        sigma_y = np.zeros((p, N))      

        P = P_escape.copy()

        for k in range(N):
            P_pred = self.A_est @ P @ self.A_est.T + Q

            if k <= self.k_prime_ctrl:
                S = self.C_est @ P_pred @ self.C_est.T + R
                K = np.linalg.solve(S.T, (P_pred @ self.C_est.T).T).T

                P = (np.eye(n) - K @ self.C_est) @ P_pred @ (np.eye(n) - K @ self.C_est).T + K @ R @ K.T
                
            else:
                P = P_pred

            Sigma_y[:, :, k] = self.C_est @ P @ self.C_est.T + R
            diags = np.diag(Sigma_y[:, :, k]).copy()
            diags[diags < 0] = 0.0  
            sigma_y[:, k] = np.sqrt(diags)

        self.R_equivalent = R
        self.Q_equivalent = Q
        self.outputCovariance = Sigma_y
        self.outputStd = sigma_y

        return sigma_y

    def computeUncertaintyWorstCase(self):
        u = np.array(self.inputValues)
        if u.ndim == 1:
            u = u.reshape(-1, 1)

        N = u.shape[1]

        G = self.buildObserverGain()

        # Parameters for worst case in training
        epsilon_escape_train = self.epsilon[:, :self.k_prime]
        R_train = (epsilon_escape_train @ epsilon_escape_train.T) / epsilon_escape_train.shape[1]
        eps_max_train = np.max(np.linalg.norm(epsilon_escape_train, axis=0))
        eps_sq_train = eps_max_train**2

        # Parameters for worst case outside training
        epsilon_escape_free = self.epsilon[:, self.k_prime : self.k_prime_ctrl + 1]
        R_free = (epsilon_escape_free @ epsilon_escape_free.T) / epsilon_escape_free.shape[1]
        eps_max_free = np.max(np.linalg.norm(epsilon_escape_free, axis=0))
        eps_sq_free = eps_max_free**2

        
        A_est = np.ascontiguousarray(self.A_est)
        C_est = np.ascontiguousarray(self.C_est)
        G = np.ascontiguousarray(G)
        R_train = np.ascontiguousarray(R_train)
        R_free = np.ascontiguousarray(R_free)

        sigma_y, Sigma_y = _compute_uncertainty_worst_case_jit(
            A_est, C_est, G, self.k_prime, self.k_prime_ctrl, N,
            eps_sq_train, eps_sq_free, R_train, R_free
        )

        self.outputStd = sigma_y
        return sigma_y
    
@njit(cache=True, fastmath=True)
def _compute_uncertainty_worst_case_jit(A_est, C_est, G, k_prime, k_prime_ctrl, N, 
                                        eps_sq_train, eps_sq_free, R_train, R_free):
    n = A_est.shape[0]
    p = C_est.shape[0]
    m = G.shape[1]
    
    Sigma_y = np.zeros((p, p, N))
    sigma_y = np.zeros((p, N))
    
    Sigma_y_accum = np.zeros((p, p))
    
    V_train = G.copy()
    G_sum_train = np.zeros((n, m))
    
    V_free = G.copy()
    G_sum_free = np.zeros((n, m))
    
    for k in range(N):
        if k <= k_prime:
            G_sum_train += V_train
            V_train = A_est @ V_train  # O(n^2 * p) -> O(n^3)
            
            M = C_est @ G_sum_train    
            Sigma_y_accum += (M @ M.T) * eps_sq_train
            
            Sigma_y[:, :, k] = Sigma_y_accum + R_train
        else:
            G_sum_free += V_free
            V_free = A_est @ V_free    #  O(n^2 * p)
            

            M = C_est @ G_sum_free     
            Sigma_y_accum += (M @ M.T) * eps_sq_free
            
            Sigma_y[:, :, k] = Sigma_y_accum + R_free
        
        for i in range(p):
            val = Sigma_y[i, i, k]
            if val < 0.0:
                val = 0.0
            sigma_y[i, k] = np.sqrt(val)
            
    return sigma_y, Sigma_y