import numpy as np
from scipy.stats import genpareto

class EVTStatisticalBound:
    def __init__(self, residuals, number_of_pending_iterations, percentile_threshold = 90, confidence_level = 0.95, tolerance = 1e-12):
        self.residuals = residuals
        self.number_of_pending_iterations = number_of_pending_iterations
        self.percentile_threshold = percentile_threshold
        self.confidence_level = confidence_level
        self.tolerance = tolerance

        self.check_errors()

    def check_errors(self):
        if len(self.residuals) == 0:
            raise ValueError("Residuals array is empty.")

        if self.number_of_pending_iterations <= 0:
            raise ValueError("Number of pending iterations must be positive.")

        if not (0 < self.percentile_threshold < 100):
            raise ValueError("Percentile threshold must be between 0 and 100.")

        if not (0 < self.confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1.")

    def get_pot_threshold(self):
        u = np.percentile(self.residuals, self.percentile_threshold)
        exceedances = self.residuals[self.residuals > u]

        if exceedances.size < 10:
            raise ValueError(
                f"Too few exceedances ({exceedances.size}) for GPD fitting. "
                "Lower percentile_threshold."
            )

        return u, exceedances

    def estimate_extremal_index(self, residuals, pot_threshold):
        # Ferro-Segers estimator for the extremal index
        # Ferro, C.A.T. and Segers, J. (2003), Inference for clusters of extreme values. Journal of the Royal Statistical Society: Series B 
        # (Statistical Methodology), 65: 545-556. https://doi.org/10.1111/1467-9868.00401
        z = np.abs(np.asarray(residuals).ravel())
        exceedance_idx = np.flatnonzero(z > pot_threshold)
        n_exceedances = len(exceedance_idx)

        if n_exceedances < 3:
            raise ValueError(
                f"Too few exceedances ({n_exceedances}) to estimate "
                "the extremal index."
            )

        T = np.diff(exceedance_idx)

        if np.max(T) <= 2:
            theta = (
                2.0 * np.sum(T)**2
                / ((n_exceedances - 1) * np.sum(T**2))
            )

        else:

            T1 = T - 1
            T2 = T - 2

            denominator = np.sum(T1 * T2)

            if denominator <= 0:
                raise ValueError(
                    "Ferro-Segers corrected estimator is undefined "
                    "for these inter-exceedance times."
                )

            theta = (
                2.0 * np.sum(T1)**2
                / ((n_exceedances - 1) * denominator)
            )

        theta = np.clip(theta, 1e-12, 1.0)

        return theta

    def estimate_individual_exceeding_probability(self, theta, lambda_u):
        return -np.log(self.confidence_level) / (theta * lambda_u * self.number_of_pending_iterations)

    def estimate_statistical_bound(self, q, xi, sigma, u):
        if np.abs(xi) < self.tolerance:
            return u - sigma * np.log(q)
        else:
            return u + (sigma / xi) * (q**(-xi) - 1)

    def compute_statistical_bound(self):
        if np.max(self.residuals) <= self.tolerance:
            return np.zeros_like(self.number_of_pending_iterations)

        u, exceedances = self.get_pot_threshold()
        theta = self.estimate_extremal_index(self.residuals, u)

        xi, loc, sigma = genpareto.fit(
            exceedances,
            floc = 0
        )

        if sigma <= 0:
            raise ValueError(
                "The fitted GPD scale parameter sigma is not positive."
            )

        lambda_u = exceedances.size / self.residuals.size
        q = self.estimate_individual_exceeding_probability(theta, lambda_u)

        if np.any(q > 1):
            raise ValueError(
                "The requested horizon/confidence level does not "
                "lead to a valid tail extrapolation: q > 1."
            )

        B = self.estimate_statistical_bound(q, xi, sigma, u)

        return B

    def get_statistical_info(self):
        u, exceedances = self.get_pot_threshold()
        theta = self.estimate_extremal_index(self.residuals, u)

        xi, loc, sigma = genpareto.fit(
            exceedances,
            floc = 0
        )

        lambda_u = exceedances.size / self.residuals.size
        n_exceedances = exceedances.size

        return {
            "POT threshold": u,
            "xi GPD": xi,
            "sigma GPD": sigma,
            "lambda_u": lambda_u,
            "theta": theta,
            "n_exceedances": n_exceedances,
        }