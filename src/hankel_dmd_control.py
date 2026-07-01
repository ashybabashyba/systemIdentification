import numpy as np
from scipy.linalg import schur

class Hankel_DMDc:
    def __init__(self, systemInput, systemOutput, energyThreshold=1-1e-9):
        self.systemInput  = systemInput
        self.systemOutput = systemOutput

        self.energyThreshold = energyThreshold
        self.numberOfOutputs = systemOutput.shape[0]

    def buildHankelMatrix(self, data):
        if data.ndim == 1:
            data = data.reshape(1, data.shape[0])
        
        M, N = data.shape
        L = N // 2

        num_rows = N - L + 1
        H = np.zeros((num_rows, M * L))

        for k in range(num_rows):
            block = data[:, k:k+L]
            H[k, :] = block.flatten(order='F')
        return H.T
    
    def build_LHS_HankelMatrix(self):
        lengths = [y.shape[0] for y in self.systemOutput]
        if len(set(lengths)) != 1:
            raise ValueError("All output signals must have the same length.")

        return self.buildHankelMatrix(self.systemOutput[:, 1:])
    
    def build_RHS_HankelMatrix(self):
        lengths = [y.shape[0] for y in self.systemOutput]
        if len(set(lengths)) != 1:
            raise ValueError("All output signals must have the same length.")

        outputHankel = self.buildHankelMatrix(self.systemOutput[:, :-1])
        inputHankel = self.buildHankelMatrix(self.systemInput[:-1])

        return np.vstack((outputHankel, inputHankel))
    
    def compute_AB_hat(self):
        LHS = self.build_LHS_HankelMatrix() 
        RHS = self.build_RHS_HankelMatrix() 
        
        L = LHS.shape[0] 

        U, s, Vh = np.linalg.svd(RHS, full_matrices=False)
        cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
        r = np.argmax(cumulative_energy >= self.energyThreshold) + 1
        
        U_r = U[:, :r]          
        s_r = s[:r]             
        Vh_r = Vh[:r, :]        
        
        inv_Sigma_r = np.diag(1.0 / s_r)
        V_r = Vh_r.T.conj()     
        
        RHS_pseudo_inv = np.dot(V_r, np.dot(inv_Sigma_r, U_r.T.conj())) 
        
        G = np.dot(LHS, RHS_pseudo_inv) 
        
        A_hat_raw = G[:, :L] # (L, L)
        B_hat_raw = G[:, L:] # (L, m)
        A_hat_stable = stabilize_schur_smooth(A_hat_raw, epsilon=1e-6)
        
        X_hankel = RHS[:L, :]  
        U_hankel = RHS[L:, :]  
        
        residuo_estado = LHS - np.dot(A_hat_stable, X_hankel)
        
        B_hat_T_stable, _, _, _ = np.linalg.lstsq(U_hankel.T, residuo_estado.T, rcond=None)
        B_hat_stable = B_hat_T_stable.T
        
        return A_hat_stable, B_hat_raw
        
    
    def evolve_Hankel_DMDc(self, A_hat, B_hat, u_completo):
        u_completo = np.atleast_2d(u_completo)
        n_steps = u_completo.shape[1]
        
        M = self.numberOfOutputs
        L = A_hat.shape[0] // M 
        
        y = np.zeros((M, n_steps))
        
        y[:, :L] = self.systemOutput[:, :L]
        
        x_hat = y[:, :L].flatten(order='F').reshape(-1, 1) 
        
        for k in range(L - 1, n_steps - 1):
            u_window = u_completo[:, k - L + 1 : k + 1]
            u_hat = u_window.flatten(order='F').reshape(-1, 1)
            
            x_next = np.dot(A_hat, x_hat) + np.dot(B_hat, u_hat)
            
            y[:, k + 1] = x_next[-M:, 0]
            

            x_hat = np.roll(x_hat, -M)
            x_hat[-M:, 0] = y[:, k + 1]
            
        return y

def stabilize_schur_smooth(A, epsilon=1e-6):
    T, Q = schur(A, output='real')
    n = A.shape[0]
    i = 0

    while i < n:
        if i == n - 1 or abs(T[i+1, i]) < 1e-12:
            lam = T[i, i]
            scale = (1 - epsilon) / max(1.0, abs(lam))
            T[i, i] = scale * lam
            i += 1
        else:
            T_block = T[i:i+2, i:i+2]
            eigvals = np.linalg.eigvals(T_block)
            r = max(abs(eigvals))
            scale = (1 - epsilon) / max(1.0, r)
            T[i:i+2, i:i+2] = scale * T_block
            i += 2

    return Q @ T @ Q.T