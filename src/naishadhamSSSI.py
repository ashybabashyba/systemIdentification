import numpy as np
import matplotlib.pyplot as plt

class StateSpace:
    def __init__(self, systemInput, systemOutput, energyThreshold=1-1e-6):
        self.systemInput  = systemInput
        self.systemOutput = systemOutput

        self.energyThreshold = energyThreshold

    def buildHankelMatrix(self):
        N = self.systemOutput.shape[0]
        L = int(N / 2)

        row_idx = np.arange(N - L + 1)[:, None]   
        col_idx = np.arange(L)[None, :]           

        idx = row_idx + col_idx                   
        return self.systemOutput[idx]        

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
    
    def buildStateSpaceSystem(self):
        omega_L, X_L = self.buildObservabilityAndStateMatrices()

        omega1 = omega_L[:-1, :]   # Observability matrix without last row L-rl
        omega2 = omega_L[1:, :]    # Observability matrix without first row L-r1

        # The paper has a typo error, see the reference 41
        omega_LHS = np.matmul(omega1.conj().T, omega1)
        omega_RHS = np.matmul(omega1.conj().T, omega2)

        # A, _, _, _ = np.linalg.lstsq(omega_LHS, omega_RHS, rcond=None)
        A, _, _, _ = np.linalg.lstsq(omega1, omega2, rcond=None)
        C = omega_L[0, :]

        r = A.shape[0]
        N = self.systemInput.shape[0]

        Omega_rows = np.zeros((N, r))
        Ak = np.eye(r)
        for k in range(N):
            Omega_rows[k, :] = np.matmul(C, Ak)
            Ak = np.matmul(Ak, A)

        Womega = np.zeros((N, 1 + r))
        Womega[:, 0] = self.systemInput.reshape(N,)

        for k in range(N):
            if k == 0:
                Womega[k, 1:] = 0.0
            else:
                past_w_reversed = self.systemInput[:k][::-1]            
                Omegas = Omega_rows[:k, :]               
                Womega[k, 1:] = np.matmul(past_w_reversed, Omegas)

        Y = self.systemOutput.reshape(N, )
        x1 = X_L[:, 0].reshape((r,))  # vector r
        RHS = Y - np.matmul(Omega_rows, x1)    # shape (N,)
        # WT_L = np.matmul(Womega.conj().T, Womega)
        # WT_R = np.matmul(Womega.conj().T, RHS)        

        theta, _, _, _ = np.linalg.lstsq(Womega, RHS, rcond=None)
        D = float(theta[0])
        B = theta[1:].reshape((r, 1))

        return A, B, C, D

    def evolveInput(self, A, B, C, D, u, x0=None):
        u = np.asarray(u).reshape(-1)       
        n_steps = len(u)
        r = A.shape[0]                      

        A = np.asarray(A)
        B = np.asarray(B).reshape(r, 1)
        C = np.asarray(C).reshape(1, r)
        D = float(D)

        x = np.zeros((n_steps, r, 1))
        y = np.zeros((n_steps, 1))

        if x0 is None:
            x[0] = np.zeros((r, 1))
        else:
            x[0] = np.asarray(x0).reshape(r, 1)

        y[0] = np.matmul(C, x[0]) + D * u[0]

        for k in range(1, n_steps):
            x[k] = np.matmul(A, x[k-1]) + B * u[k-1]
            y[k] = np.matmul(C, x[k]) + D * u[k]

        return x, y