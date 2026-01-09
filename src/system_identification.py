import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import schur

class StateSpace:
    def __init__(self, systemInput, systemOutput, energyThreshold=1-1e-6):
        self.systemInput  = systemInput
        self.systemOutput = systemOutput

        self.energyThreshold = energyThreshold
        self.numberOfOutputs = systemOutput.shape[0]

    def buildHankelMatrix(self):
        for i in range(self.numberOfOutputs - 1):
            if self.systemOutput[i].shape[0] != self.systemOutput[i + 1].shape[0]:
                raise ValueError("All output signals must have the same length.")
            
        M, N = self.systemOutput.shape
        L = N // 2

        num_rows = N - L + 1           # columnas de la hankel

        H = np.zeros((num_rows, M * L))

        for k in range(num_rows):
            block = self.systemOutput[:, k:k+L]        
            H[k, :] = block.flatten(order='F')  

        return H.T
    
    def buildInputHankelMatrix(self):
        N = self.systemInput.shape[0]
        L = int(N/2)

        num_rows = N - L + 1

        H = np.zeros((num_rows, L))

        for k in range(num_rows):
            H[k, :] = self.systemInput[k:k+L]

        return H.T

    def energyCriterionForTruncation(self, singularValues, order=2):
        total_energy = np.sum(singularValues**order)
        cumulative_energy = np.cumsum(singularValues**order) / total_energy
        r = np.searchsorted(cumulative_energy, self.energyThreshold) + 1
        return r                     
    
    def buildObservabilityAndStateMatrices(self):
        H = self.buildHankelMatrix()

        U, S, Vh = np.linalg.svd(H, full_matrices=False)
        r = self.energyCriterionForTruncation(S)

        Ur = U[:, :r]
        Sr = S[:r]
        Vhr = Vh[:r, :]

        S_sqrt = np.diag(np.sqrt(Sr))

        observabilityMatrix = np.matmul(Ur, S_sqrt)
        stateMatrix = np.matmul(S_sqrt, Vhr)

        return observabilityMatrix, stateMatrix
    
    def buildInputOutputCorrelationMatrix(self):
        H_y = self.buildHankelMatrix()
        H_u = self.buildInputHankelMatrix()

        number_of_columns = H_u.shape[1]

        R_yy = np.matmul(H_y, H_y.T) / number_of_columns
        R_yu = np.matmul(H_y, H_u.T) / number_of_columns
        R_uu = np.matmul(H_u, H_u.T) / number_of_columns
        R_uy = np.matmul(H_u, H_y.T) / number_of_columns

        R_hh = R_yy - np.matmul(np.matmul(R_yu, np.linalg.pinv(R_uu)), R_uy)

        return R_hh
    
    def buildObservabilityMatrix_juang(self):
        R_hh = self.buildInputOutputCorrelationMatrix()

        U, S, Vh = np.linalg.svd(R_hh, full_matrices=False)
        r = self.energyCriterionForTruncation(S)

        Ur = U[:, :r]
        Sr = S[:r]
        Vhr = Vh[:r, :]

        S_sqrt = np.diag(np.sqrt(Sr))

        observabilityMatrix = Ur

        return observabilityMatrix
    
    def buildStateSpaceSystem(self):
        # omega_L, X_L = self.buildObservabilityAndStateMatrices()  # For naishadham method
        omega_L = self.buildObservabilityMatrix_juang()

        omega1 = omega_L[:-self.numberOfOutputs, :]   # Observability matrix without last row L-rl
        omega2 = omega_L[self.numberOfOutputs:, :]    # Observability matrix without first row L-r1

        A, _, _, _ = np.linalg.lstsq(omega1, omega2, rcond=None)
        A = stabilize_matrix(A) 
        C, _, _, _ = np.linalg.lstsq(A.T, omega_L[self.numberOfOutputs, :].T, rcond=None)

        r = A.shape[0]
        N = self.systemInput.shape[0]

        Omega_rows = np.zeros((self.numberOfOutputs*N, r))
        Ak = np.eye(r)
        for k in range(N):
            Omega_rows[self.numberOfOutputs*k:self.numberOfOutputs*(k+1), :] = np.matmul(C, Ak)
            Ak = np.matmul(Ak, A)

        Womega = np.zeros((self.numberOfOutputs*N, self.numberOfOutputs + r))
        auxiliaryIdentity = np.eye(self.numberOfOutputs)
        for k in range(N):
            Womega[self.numberOfOutputs*k:self.numberOfOutputs*(k+1), :self.numberOfOutputs] = auxiliaryIdentity * self.systemInput[k]

        for k in range(N):
            if k == 0:
                Womega[self.numberOfOutputs*k:self.numberOfOutputs*(k+1), self.numberOfOutputs:] = 0.0
            else:
                past_w_reversed = self.systemInput[:k][::-1]            
                Omegas = Omega_rows[:self.numberOfOutputs*k, :]               
                Womega[self.numberOfOutputs*k:self.numberOfOutputs*(k+1), self.numberOfOutputs:] = np.matmul(past_w_reversed, Omegas)

        M, N = self.systemOutput.shape
        Y = self.systemOutput.T.reshape(M * N,)
        initialState = np.zeros((r,))
        # initialState = X_L[:, 0].reshape((r,))  # For naishadham method
        RHS = Y - np.matmul(Omega_rows, initialState)    

        theta, _, _, _ = np.linalg.lstsq(Womega, RHS, rcond=None)
        D = theta[:self.numberOfOutputs].reshape((self.numberOfOutputs, 1))
        B = theta[self.numberOfOutputs:].reshape((r, 1))

        return A, B, C, D, initialState

    def evolveInput(self, A, B, C, D, u, x0):
        u = np.asarray(u).reshape(-1)       
        n_steps = len(u)
        r = A.shape[0]                      

        A = np.asarray(A)
        B = np.asarray(B).reshape(r, 1)
        C = np.asarray(C).reshape(self.numberOfOutputs, r)
        D = np.asarray(D).reshape(self.numberOfOutputs, 1)

        x = np.zeros((n_steps, r, 1))
        y = np.zeros((self.numberOfOutputs, n_steps))

        x[0] = np.asarray(x0).reshape(r, 1)
        y[0] = np.matmul(C, x[0]) + D * u[0]

        for k in range(1, n_steps):
            x[k] = np.matmul(A, x[k-1]) + B * u[k-1]
            y[:, k] = np.matmul(C, x[k]) + D * u[k]

        return x, y
    
def stabilize_matrix(A, epsilon=1e-6):
    T, Q = schur(A, output='real')
    n = A.shape[0]
    i = 0

    while i < n:
        if i == n - 1 or abs(T[i+1, i]) < 1e-12:
            lam = T[i, i]
            if abs(lam) >= 1:
                T[i, i] = lam / (abs(lam) + epsilon)
            i += 1
        else:
            T_block = T[i:i+2, i:i+2]
            eigvals = np.linalg.eigvals(T_block)
            r = max(abs(eigvals))

            if r >= 1:
                scale = (1 - epsilon) / r  
                T[i:i+2, i:i+2] = scale * T_block
            i += 2

    A_stable = Q @ T @ Q.T
    return A_stable