import numpy as np
from scipy.signal import dlsim
from numba import njit

class VicarioOKID:
    def __init__(
        self,
        A,
        B,
        C,
        D,
        initialState,
        dataWrapper,
        epsilon=None
    ):
        dataWrapper.buildInterpolatedOutputs()
        self.k_prime = dataWrapper.k_prime
        self.k_prime_ctrl = dataWrapper.k_prime_ctrl
        self.inputValues = dataWrapper.inputValues
        self.y_ref_interp = dataWrapper.interpolatedReferenceOutputValues

        if epsilon is None:
            self.residuals = dataWrapper.buildOutputResiduals()
        else:
            eps = np.array(epsilon)
            if eps.ndim == 1:
                eps = eps.reshape(1, -1)

            if eps.shape[1] < (self.k_prime_ctrl + 1):
                raise ValueError(
                    "Provided epsilon must cover at least the control window "
                    f"(required length >= {self.k_prime_ctrl + 1}, got {eps.shape[1]})."
                )

            self.residuals = eps

        self.extractSystemMatrices(A, B, C, D)
        self.initialState = initialState

    def extractSystemMatrices(self, A, B, C, D):
        u = np.array(self.inputValues)
        if u.ndim == 1:
            u = u.reshape(1, -1)

        q = C.shape[0]
        m = u.shape[0]

        if B.shape[1] < m + q or D.shape[1] < m + q:
            raise ValueError(
                "Vicario model must have at least m+q input columns in B and D "
                f"(got B:{B.shape}, D:{D.shape}, expected columns>={m + q})."
            )

        self.A_est = A
        self.B_u = B[:, :m]
        self.C_est = C
        self.D_u = D[:, :m]
        self.D_e = D[:, m:m+q]
        self.K_gain = B[:, m:m+q]

    def evolveWithFilter(self):
            u = np.array(self.inputValues)
            if u.ndim == 1:
                u = u.reshape(1, -1)
    
            N = u.shape[1]
            n = self.A_est.shape[0]
            p = self.C_est.shape[0]
    
            x = np.zeros((n, N))
            y = np.zeros((p, N))   
    
            epsilon = self.residuals
    
            x[:, 0] = self.initialState
            y[:, 0] = self.C_est @ x[:, 0] + self.D_u @ u[:,0] 
    
            for k in range(1, N):
                x_pred = self.A_est @ x[:, k-1] + self.B_u @ u[:,k-1]
                y_pred = self.C_est @ x_pred + self.D_u @ u[:,k]
    
                if k <= self.k_prime_ctrl:
                    x[:, k] = x_pred + self.K_gain @ epsilon[:, k-1]
                    y[:, k] = self.C_est @ x[:, k] + self.D_u @ u[:,k] + epsilon[:, k-1]
    
                else:
                    x[:, k] = x_pred
                    y[:, k] = y_pred
    
            self.epsilon = epsilon              
    
            return x, y

    def computeUncertaintyFromModelDifference(
        self,
        A_nom,
        B_nom,
        C_nom,
        D_nom,
        A_corr,
        B_corr,
        C_corr,
        D_corr,
        inputValues=None,
        correctionInput=None,
        correctionMode='measured',
        correctionBoundWindow='train_ctrl',
        x0_nom=None,
        x0_corr=None
    ):
            if inputValues is None:
                u = np.array(self.inputValues)
            else:
                u = np.array(inputValues)

            if u.ndim == 1:
                u = u.reshape(1, -1)

            if correctionInput is None:
                eps = None
            else:
                eps = np.array(correctionInput)
                if eps.ndim == 1:
                    eps = eps.reshape(1, -1)
                eps = self._prepareCorrectionInputForWindow(
                    eps=eps,
                    target_len=u.shape[1],
                    mode=correctionMode,
                    bound_window=correctionBoundWindow
                )

            A_nom = np.ascontiguousarray(np.array(A_nom))
            B_nom = np.ascontiguousarray(np.array(B_nom))
            C_nom = np.ascontiguousarray(np.array(C_nom))
            D_nom = np.ascontiguousarray(np.array(D_nom))

            A_corr = np.ascontiguousarray(np.array(A_corr))
            B_corr = np.ascontiguousarray(np.array(B_corr))
            C_corr = np.ascontiguousarray(np.array(C_corr))
            D_corr = np.ascontiguousarray(np.array(D_corr))

            if B_nom.shape[1] != u.shape[0] or B_corr.shape[1] != u.shape[0]:
                raise ValueError(
                    "Input channel mismatch: expected number of input rows in u "
                    f"to match B_nom/B_corr columns ({B_nom.shape[1]}, {B_corr.shape[1]}), got {u.shape[0]}."
                )

            if C_nom.shape[0] != C_corr.shape[0]:
                raise ValueError(
                    "Nominal and corrected models must have the same number of outputs "
                    f"(got {C_nom.shape[0]} and {C_corr.shape[0]})."
                )

            if eps is not None and eps.shape[0] != C_corr.shape[0]:
                raise ValueError(
                    "Correction input dimension must match corrected-model outputs "
                    f"for direct epsilon feedthrough (got eps rows={eps.shape[0]}, outputs={C_corr.shape[0]})."
                )

            if eps is None:
                sigma_model, delta_y, delta_h = _compute_markov_difference_uncertainty_jit(
                    A_nom, B_nom, C_nom, D_nom,
                    A_corr, B_corr, C_corr, D_corr,
                    np.ascontiguousarray(u)
                )
            else:
                sigma_model = _compute_markov_difference_uncertainty_with_correction_jit(
                    A_nom, B_nom, C_nom, D_nom,
                    A_corr, B_corr, C_corr, D_corr,
                    np.ascontiguousarray(self.K_gain),
                    np.ascontiguousarray(u),
                    np.ascontiguousarray(eps),
                    self.k_prime_ctrl,
                    np.max(np.linalg.norm(eps[:, :self.k_prime_ctrl + 1], axis=0))
                )

            self.outputStdModelDifference = sigma_model

            return sigma_model

    def _prepareCorrectionInputForWindow(self, eps, target_len, mode, bound_window):
            if mode not in ('measured', 'max_after_control'):
                raise ValueError(
                    "correctionMode must be 'measured' or 'max_after_control'."
                )

            if bound_window not in ('train', 'train_ctrl'):
                raise ValueError(
                    "correctionBoundWindow must be 'train' or 'train_ctrl'."
                )

            if eps.shape[1] < 1:
                raise ValueError("Correction input must have at least one sample.")

            if mode == 'measured':
                if eps.shape[1] != target_len:
                    raise ValueError(
                        "Correction input length must match the input sequence length "
                        f"({eps.shape[1]} vs {target_len}) when correctionMode='measured'."
                    )
                return eps

            # max_after_control: exact epsilon up to control window, bounded beyond.
            if eps.shape[1] < (self.k_prime_ctrl + 1):
                raise ValueError(
                    "Correction input must cover at least control window when "
                    "correctionMode='max_after_control'."
                )

            eps_out = np.zeros((eps.shape[0], target_len))
            n_copy = min(eps.shape[1], target_len)
            eps_out[:, :n_copy] = eps[:, :n_copy]

            if bound_window == 'train':
                end_idx = self.k_prime
            else:
                end_idx = self.k_prime_ctrl

            end_idx = min(end_idx, eps.shape[1] - 1)
            if end_idx < 0:
                raise ValueError("Invalid bound window for epsilon statistics.")

            eps_bound = np.max(np.abs(eps[:, :end_idx + 1]), axis=1)

            start_extrap = self.k_prime_ctrl + 1
            if start_extrap < target_len:
                for row in range(eps_out.shape[0]):
                    sign = 1.0
                    if start_extrap > 0 and np.abs(eps_out[row, start_extrap - 1]) > 0.0:
                        sign = np.sign(eps_out[row, start_extrap - 1])
                    eps_out[row, start_extrap:] = sign * eps_bound[row]

            return eps_out

@njit(cache=True, fastmath=True)
def _compute_markov_difference_uncertainty_jit(
    A_nom, B_nom, C_nom, D_nom,
    A_corr, B_corr, C_corr, D_corr,
    u
):
    N = u.shape[1]
    p = C_nom.shape[0]
    m = u.shape[0]

    # Delta Markov parameters for channel u -> y
    delta_h = np.zeros((p, m, N))
    delta_h[:, :, 0] = D_corr - D_nom

    phi_nom = B_nom.copy()
    phi_corr = B_corr.copy()

    for i in range(1, N):
        delta_h[:, :, i] = C_corr @ phi_corr - C_nom @ phi_nom
        phi_nom = A_nom @ phi_nom
        phi_corr = A_corr @ phi_corr

    # Deterministic discrepancy on known input: delta_y(k) = sum_i DeltaH_i u(k-i)
    delta_y = np.zeros((p, N))
    sigma_model = np.zeros((p, N))

    for k in range(N):
        accum = np.zeros(p)
        for i in range(k + 1):
            accum += delta_h[:, :, i] @ u[:, k - i]

        delta_y[:, k] = accum
        for j in range(p):
            sigma_model[j, k] = np.abs(accum[j])

    return sigma_model, delta_y, delta_h


@njit(cache=True, fastmath=True)
def _compute_markov_difference_uncertainty_with_correction_jit(
    A_nom, B_nom, C_nom, D_nom,
    A_corr, B_corr, C_corr, D_corr,
    B_corr_e,   
    u, eps,
    k_train_end,
    eps_max
):
    p, m_u = C_nom.shape[0], u.shape[0]
    m_e = eps.shape[0]
    N = u.shape[1]
    m_tot = m_u + m_e

    k_train_end_val = k_train_end
    eps_max_val = eps_max

    delta_h = np.empty((p, m_tot, N), dtype=u.dtype)

    delta_h[:, :m_u, 0] = D_corr - D_nom
    delta_h[:, m_u:, 0] = 0.0
    for j in range(min(p, m_e)):
        delta_h[j, m_u + j, 0] = 1.0

    phi_nom = B_nom.copy()
    phi_corr_u = B_corr.copy()
    phi_corr_e = B_corr_e.copy()   

    for i in range(1, N):
        delta_h[:, :m_u, i] = C_corr @ phi_corr_u - C_nom @ phi_nom
        delta_h[:, m_u:, i] = C_corr @ phi_corr_e  

        phi_nom = A_nom @ phi_nom
        phi_corr_u = A_corr @ phi_corr_u
        phi_corr_e = A_corr @ phi_corr_e

    sigma_model = np.zeros((p, N), dtype=u.dtype)

    for k in range(N):
        d_k = np.zeros(p, dtype=u.dtype)
        
        sum_abs_H_e = np.zeros(p, dtype=u.dtype)

        for i in range(k + 1):
            t_past = k - i 
            
            d_k += delta_h[:, :m_u, i] @ u[:, t_past]
            
            if t_past <= k_train_end_val:
                d_k += delta_h[:, m_u:, i] @ eps[:, t_past]
            else:
                H_e = delta_h[:, m_u:, i]
                for j in range(p):
                    row_abs_sum = 0.0
                    for col in range(m_e):
                        row_abs_sum += np.abs(H_e[j, col])
                    sum_abs_H_e[j] += row_abs_sum

        for j in range(p):
            bound_extrap_j = sum_abs_H_e[j] * eps_max_val
            sigma_model[j, k] = np.abs(d_k[j]) + bound_extrap_j + eps_max_val

    return sigma_model

