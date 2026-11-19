import numpy as np
import pytest
import os
import sys

try:
    from src.evt_statistical_bound import EVTStatisticalBound
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
    from evt_statistical_bound import EVTStatisticalBound


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"residuals": []}, "Residuals array is empty"),
        ({"number_of_pending_iterations": 0}, "must be positive"),
        ({"percentile_threshold": 0}, "between 0 and 100"),
        ({"confidence_level": 1}, "between 0 and 1"),
    ],
)
def test_constructor_rejects_invalid_parameters(kwargs, message):
    parameters = {
        "residuals": np.arange(1.0, 31.0),
        "number_of_pending_iterations": 10,
        "percentile_threshold": 90,
        "confidence_level": 0.95,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        EVTStatisticalBound(**parameters)


def test_get_pot_threshold_requires_at_least_ten_exceedances():
    evt = EVTStatisticalBound(
        residuals=np.arange(1.0, 11.0),
        number_of_pending_iterations=10,
        percentile_threshold=90,
    )

    with pytest.raises(ValueError, match="Too few exceedances"):
        evt.get_pot_threshold()


def test_extremal_index_uses_corrected_estimator_and_clips_to_unit_interval():
    evt = EVTStatisticalBound(
        residuals=np.arange(120.0),
        number_of_pending_iterations=10,
    )
    exceedance_indices = np.array([0, 3, 13, 113])
    residuals = np.zeros(120)
    residuals[exceedance_indices] = 2.0

    theta = evt.estimate_extremal_index(residuals, pot_threshold=1.0)

    assert theta == pytest.approx(24200 / 29328)

    consecutive_exceedances = np.zeros(5)
    consecutive_exceedances[:4] = 2.0
    assert evt.estimate_extremal_index(
        consecutive_exceedances, pot_threshold=1.0
    ) == 1.0


def test_extremal_index_requires_three_exceedances():
    evt = EVTStatisticalBound(
        residuals=np.arange(1.0, 31.0),
        number_of_pending_iterations=10,
    )

    with pytest.raises(ValueError, match="Too few exceedances"):
        evt.estimate_extremal_index(np.array([0.0, 2.0, 0.0, 3.0]), 1.0)


def test_probability_and_gpd_bound_formulas():
    evt = EVTStatisticalBound(
        residuals=np.arange(1.0, 31.0),
        number_of_pending_iterations=20,
        confidence_level=0.9,
    )

    q = evt.estimate_individual_exceeding_probability(theta=0.5, lambda_u=0.25)
    assert q == pytest.approx(-np.log(0.9) / (0.5 * 0.25 * 20))
    assert evt.estimate_statistical_bound(q=0.2, xi=0.0, sigma=2.0, u=3.0) == pytest.approx(
        3.0 - 2.0 * np.log(0.2)
    )
    assert evt.estimate_statistical_bound(
        q=0.2, xi=0.4, sigma=2.0, u=3.0
    ) == pytest.approx(3.0 + (2.0 / 0.4) * (0.2 ** -0.4 - 1))


def test_zero_residuals_produce_zero_bound_without_fitting_tail():
    evt = EVTStatisticalBound(
        residuals=np.zeros(20),
        number_of_pending_iterations=10,
    )

    assert evt.compute_statistical_bound() == 0
