import numpy as np

class FilterIdentificationWrapper:
    def __init__(self):
        self.inputValues = []

        self.referenceOutputValues = []
        self.interpolatedReferenceOutputValues = []

        self.deterministicOutputValues = []
        self.interpolatedDeterministicOutputValues = []

        self.residualValues = []
        self.interpolatedResidualValues = []

        self.inputTimeData  = []
        self.deterministicOutputTimeData = []
        self.referenceOutputTimeData = []
        self.residualTimeData = []
        
        self.t_training = None
        self.t_control = None
        self.k_prime = None
        self.k_prime_ctrl = None

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

    def addResidualData(self, residualData, residualTimeData):
        if len(self.residualValues) == 0:
            self.residualValues = np.array([residualData])
        else:
            self.residualValues = np.vstack((self.residualValues, residualData))

        if len(self.residualTimeData) != 0:
            if not np.array_equal(self.residualTimeData, residualTimeData):
                raise ValueError("Residual time data must be consistent across calls")
        else:
            self.residualTimeData = residualTimeData

    def setTimeWindow(self, t_training, t_control):
        """Set training and control time windows with validation.
        
        Args:
            t_training: End time of training period
            t_control: End time of control/validation period
            
        Raises:
            ValueError: If reference data does not extend to at least t_control
        """
        if len(self.referenceOutputTimeData) == 0:
            raise ValueError("Reference output data must be added before setting time windows")
        
        t_ref_max = self.referenceOutputTimeData[-1]
        if t_ref_max < t_control:
            raise ValueError(
                f"Reference output time data must extend to at least t_control ({t_control}), "
                f"but only extends to {t_ref_max}"
            )
        
        self.t_training = t_training
        self.t_control = t_control

    def buildInterpolatedOutputs(self):
        # Validate that time windows have been set
        if self.t_training is None or self.t_control is None:
            raise ValueError("Time windows must be set using setTimeWindow() before calling buildInterpolatedOutputs()")
        
        # Calculate k_prime and k_prime_ctrl from specified time windows
        self.k_prime = np.searchsorted(self.inputTimeData, self.t_training, side='right') - 1
        self.k_prime_ctrl = np.searchsorted(self.inputTimeData, self.t_control, side='right') - 1

        # Use the maximum index for interpolation
        k_max = max(self.k_prime, self.k_prime_ctrl)
        t_common = self.inputTimeData[:k_max + 1]

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
        if len(self.residualValues) != 0:
            k_max = max(self.k_prime, self.k_prime_ctrl)
            t_common = self.inputTimeData[:k_max + 1]

            residuals = np.array(self.residualValues)
            if residuals.ndim == 1:
                residuals = residuals.reshape(-1, 1)

            residual_interp = []
            for i in range(residuals.shape[0]):
                residual_interp.append(
                    np.interp(
                        t_common,
                        self.residualTimeData,
                        residuals[i, :]
                    )
                )

            self.interpolatedResidualValues = np.vstack(residual_interp)
            return self.interpolatedResidualValues

        return self.interpolatedReferenceOutputValues - self.interpolatedDeterministicOutputValues

    def buildVicarioPreprocessedYhatAndResiduals(self, u, y, p, rcond=1e-10):
        """
        Build y_hat and residuals epsilon using an observer-like regression:
            y(k) ~= Ybar * v(k)
        with
            v(k) = [u(k), u(k-1), ..., u(k-p), y(k-1), ..., y(k-p)]

        Args:
            u: shape (m, N)
            y: shape (q, N)
            p: observer horizon/order
            rcond: pseudo-inverse cutoff

        Returns:
            y_hat: shape (q, N)
            epsilon: shape (q, N)
            ybar: shape (q, m*(p+1) + q*p)
        """
        if u.ndim == 1:
            u = u.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        m, n_u = u.shape
        q, n_y = y.shape
        if n_u != n_y:
            raise ValueError(
                "Input/output length mismatch in preprocessed stage: "
                f"{n_u} vs {n_y}"
            )

        n = n_u
        if p <= 0 or p >= n:
            raise ValueError(f"Invalid p={p}. Must satisfy 0 < p < N={n}")

        n_cols = n - p
        reg_dim = m * (p + 1) + q * p

        v_mat = np.zeros((reg_dim, n_cols))
        y_mat = np.zeros((q, n_cols))

        for idx, k in enumerate(range(p, n)):
            u_stack = [u[:, k - j] for j in range(p + 1)]
            y_stack = [y[:, k - j] for j in range(1, p + 1)]

            v_k = np.concatenate(u_stack + y_stack, axis=0)
            v_mat[:, idx] = v_k
            y_mat[:, idx] = y[:, k]

        vv_t = v_mat @ v_mat.T
        ybar = y_mat @ v_mat.T @ np.linalg.pinv(vv_t, rcond=rcond)
        y_hat_cols = ybar @ v_mat

        y_hat = np.zeros_like(y)
        y_hat[:, :p] = y[:, :p]
        y_hat[:, p:] = y_hat_cols

        epsilon = y - y_hat
        return y_hat, epsilon, ybar