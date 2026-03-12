import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import schur

from scipy.signal import dlsim

class StateSpace:
    def __init__(self, systemInput, systemOutput, truncationThreshold = 1e-9, observabilityMethod='Projection'):
        self.systemInput  = systemInput
        self.systemOutput = systemOutput

        self.truncationThreshold = truncationThreshold
        self.numberOfOutputs = systemOutput.shape[0]

        self.observabilityMethod = observabilityMethod

        if self.observabilityMethod not in ['Naishadham', 'Juang', 'Projection']:
            raise ValueError("Invalid observability method. Choose 'Naishadham', 'Juang', or 'Projection'.")

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

    def buildOutputHankelMatrix(self):
        lengths = [y.shape[0] for y in self.systemOutput]
        if len(set(lengths)) != 1:
            raise ValueError("All output signals must have the same length.")

        return self.buildHankelMatrix(self.systemOutput)
    
    def buildInputHankelMatrix(self):
        return self.buildHankelMatrix(self.systemInput)
    
    def buildTruncatedSVD(self, matrix):
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        r = np.sum(S > self.truncationThreshold*S[0])

        return U[:, :r], S[:r], Vh[:r, :]

    def buildObservability_Naishadham(self, HankelOutput):
        Ur, Sr, Vhr = self.buildTruncatedSVD(HankelOutput)
        S_sqrt = np.diag(np.sqrt(Sr))

        observabilityMatrix = np.matmul(Ur, S_sqrt)
        stateMatrix = np.matmul(S_sqrt, Vhr)

        return observabilityMatrix, stateMatrix
    
    def buildResidualCovarianceMatrix(self, HankelInput, HankelOutput):
        numberOfColumns = HankelInput.shape[1]

        # In problems with poor input excitation, the input Hankel matrix can be rank-deficient.
        # To mitigate this, we need to exclude the near-zero singular values
        U, S, Vh = np.linalg.svd(HankelInput, full_matrices=False)
        nonzero_indices = S > 0

        U1 = U[:, nonzero_indices]
        S1 = S[nonzero_indices]
        V1 = Vh[nonzero_indices, :]
        H_u_reduced = U1 @ np.diag(S1) @ V1

        R_yy = np.matmul(HankelOutput, HankelOutput.T) / numberOfColumns
        R_yu = np.matmul(HankelOutput, H_u_reduced.T) / numberOfColumns
        R_uu = np.matmul(H_u_reduced, H_u_reduced.T) / numberOfColumns
        R_uy = np.matmul(H_u_reduced, HankelOutput.T) / numberOfColumns

        return R_yy - np.matmul(np.matmul(R_yu, np.linalg.pinv(R_uu, rcond=1e-24)), R_uy)
    
    def buildObservability_Juang(self, HankelInput, HankelOutput):
        R_hh = self.buildResidualCovarianceMatrix(HankelInput, HankelOutput)
        U, S, Vh = self.buildTruncatedSVD(R_hh)

        return U
    
    def buildProjectionOperators(self, HankelInput):
        _, S, Vh = np.linalg.svd(HankelInput, full_matrices=False)
        r = np.sum(S > self.truncationThreshold*S[0])

        orthogonal_operator = Vh[r:, :].T @ Vh[r:, :]
        parallel_operator = Vh[:r, :].T @ Vh[:r, :]

        return orthogonal_operator, parallel_operator

    
    def buildObservability_Projection(self):
        Hy = self.buildOutputHankelMatrix()
        Hu = self.buildInputHankelMatrix()

        orthogonal_operator, parallel_operator = self.buildProjectionOperators(Hu)

        U_orthogonal, _, _ = self.buildTruncatedSVD(Hy @ orthogonal_operator)
        U_parallel = self.buildObservability_Juang(Hu @ parallel_operator, Hy @ parallel_operator)
        U_combined = np.hstack((U_parallel, U_orthogonal))

        return self.buildTruncatedSVD(U_combined)[0]
    
    def buildObservability_and_InitialState(self):
        if self.observabilityMethod == 'Naishadham':
            observability, X_L = self.buildObservability_Naishadham(self.buildOutputHankelMatrix())
            initialState = X_L[:, 0].reshape((observability.shape[1],))  
        elif self.observabilityMethod == 'Juang':
            observability = self.buildObservability_Juang(self.buildInputHankelMatrix(), self.buildOutputHankelMatrix())
            initialState = np.zeros((observability.shape[1],))
        elif self.observabilityMethod == 'Projection':
            observability = self.buildObservability_Projection()
            initialState = np.zeros((observability.shape[1],))
        else:
            raise ValueError("Invalid observability method. Choose 'Naishadham', 'Juang', or 'Projection'.")

        return observability, initialState
    
    def build_A_C_matrices(self, observabilityMatrix):
        omega1 = observabilityMatrix[:-self.numberOfOutputs, :]
        omega2 = observabilityMatrix[self.numberOfOutputs:, :]

        A, _, _, _ = np.linalg.lstsq(omega1, omega2, rcond=None)
        A = stabilize_schur_smooth(A)

        C, _, _, _ = np.linalg.lstsq(A.T, observabilityMatrix[self.numberOfOutputs:2*self.numberOfOutputs, :].T, rcond=None)
        C = np.asarray(C).reshape(self.numberOfOutputs, A.shape[0])

        return A, C
    
    def buildWeightedObserability(self, A, C):
        r = A.shape[0]
        M, N = self.systemOutput.shape
        n_out = self.numberOfOutputs

        Omega_rows = np.zeros((n_out * N, r))
        
        current_CA = C
        for k in range(N):
            Omega_rows[k*n_out : (k+1)*n_out, :] = current_CA
            current_CA = current_CA @ A  

        Womega = np.zeros((n_out * N, n_out + r))

        inputs_expanded = np.repeat(self.systemInput, n_out).reshape(-1, 1)
        Womega[:, :n_out] = np.tile(np.eye(n_out), (N, 1)) * inputs_expanded

        if n_out == 1:
            input_steps = np.zeros((N, N))
            for i in range(1, N):
                input_steps[i, :i] = self.systemInput[:i][::-1]
            
            Womega[:, n_out:] = input_steps @ Omega_rows
        else:
            for k in range(1, N):
                past_w = self.systemInput[:k][::-1]
                CA_blocks = Omega_rows[:k*n_out].reshape(k, n_out, r)
                block = np.tensordot(past_w, CA_blocks, axes=(0, 0))

                Womega[k*n_out:(k+1)*n_out, n_out:] = block

        return Omega_rows, Womega
    
    def build_B_D_matrices(self, A, C, initialState):
        Omega_rows, Womega = self.buildWeightedObserability(A, C)
        M, N = self.systemOutput.shape
        Y = self.systemOutput.reshape(-1, order="F")

        if not np.any(initialState):
            RHS = Y
        else:
            RHS = Y - Omega_rows @ initialState

        theta, _, _, _ = np.linalg.lstsq(Womega, RHS, rcond=None)
        D = theta[:self.numberOfOutputs].reshape((self.numberOfOutputs, 1))
        B = theta[self.numberOfOutputs:].reshape((A.shape[0], 1))

        return B, D
    
    def buildStateSpaceSystem(self):
        observabilityMatrix, initialState = self.buildObservability_and_InitialState()
        A, C = self.build_A_C_matrices(observabilityMatrix)
        B, D = self.build_B_D_matrices(A, C, initialState)

        return A, B, C, D, initialState
    

    def evolveInput(self, A, B, C, D, u, x0):
        system = (A, B, C, D, 1.0)
        t, y, x = dlsim(system, u, x0=x0)
        
        return x, y.T 
    
def stabilize_matrix(A, epsilon=1e-9):
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

def stabilize_schur_smooth(A, epsilon=1e-6):
    T, Q = schur(A, output='real')
    n = T.shape[0]
    limit = 1.0 - epsilon

    diag_indices = np.diag_indices(n)
    lams = T[diag_indices]
    
    bad_lams_mask = np.abs(lams) >= 1.0
    if np.any(bad_lams_mask):
        scales = limit / np.abs(lams[bad_lams_mask])
        T[diag_indices[0][bad_lams_mask], diag_indices[1][bad_lams_mask]] *= scales

    subdiag = np.diag(T, k=-1)
    idx_2x2 = np.where(np.abs(subdiag) > 1e-12)[0]

    for i in idx_2x2:
        T_block = T[i:i+2, i:i+2]
        r_block = np.max(np.abs(np.linalg.eigvals(T_block)))
        
        if r_block >= 1.0:
            scale = limit / r_block
            T[i:i+2, i:i+2] *= scale

    return Q @ T @ Q.T