import numpy as np
import pytest
from scipy.stats import genpareto

try:
    from src.evt_statistical_bound import EVTStatisticalBound
except ImportError:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
    from evt_statistical_bound import EVTStatisticalBound


def test_synthetic_negative_shape_gpd_fit_and_endpoint_warning():
    rng = np.random.default_rng(2026)
    excesses = genpareto.rvs(
        c=-0.2,
        loc=0,
        scale=2,
        size=9999,
        random_state=rng,
    )
    residuals = np.concatenate(
        (
            np.linspace(0, 9.9, 80000),
            np.full(10002, 10.0),
            10.0 + excesses,
        )
    )
    evt = EVTStatisticalBound(
        residuals=residuals,
        number_of_pending_iterations=100,
        percentile_threshold=90,
    )

    info = evt.get_statistical_info()
    assert info["u"] == pytest.approx(10)
    assert info["xi"] == pytest.approx(-0.2, abs=0.03)
    assert info["sigma"] == pytest.approx(2, abs=0.1)
    assert info["n_exceedances"] == excesses.size
    assert info["endpoint"] == pytest.approx(10 - info["sigma"] / info["xi"])

    with pytest.warns(RuntimeWarning, match="finite endpoint") as warning_record:
        bound = evt.compute_statistical_bound()

    assert bound < info["endpoint"]
    assert f"horizon H={evt.number_of_pending_iterations}" in str(warning_record[0].message)
    assert f"{np.max(np.abs(residuals)):.6g}" in str(warning_record[0].message)


def test_pot_threshold_and_rate_are_symmetric_for_signed_residuals():
    magnitudes = np.repeat(np.arange(1.0, 101.0), 2)
    residuals = np.concatenate((magnitudes, -magnitudes))
    evt = EVTStatisticalBound(
        residuals=residuals,
        number_of_pending_iterations=100,
        percentile_threshold=90,
    )

    u, exceedances = evt.get_pot_threshold()
    assert u == pytest.approx(90.1)
    np.testing.assert_allclose(
        np.sort(exceedances),
        np.repeat(np.arange(0.9, 10.0, 1.0), 4),
    )

    info = evt.get_statistical_info()
    assert info["lambda_u"] == pytest.approx(0.1)
    assert info["n_exceedances"] == 40
