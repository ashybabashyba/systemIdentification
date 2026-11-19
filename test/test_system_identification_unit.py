import numpy as np
import pytest
from scipy.signal import dlsim
import os
import sys

try:
    from src.system_identification import (
        StateSpace,
        stabilize_matrix,
        stabilize_schur_smooth,
    )
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
    from system_identification import (
        StateSpace,
        stabilize_matrix,
        stabilize_schur_smooth,
    )


def make_state_space(system_input=None, system_output=None, **kwargs):
    if system_input is None:
        system_input = np.array([0.0, 1.0, -0.5, 0.25, 0.75])
    if system_output is None:
        system_output = np.zeros((1, len(system_input)))
    return StateSpace(system_input, system_output, **kwargs)


def test_constructor_normalizes_single_input_and_rejects_unknown_method():
    state_space = make_state_space()

    assert state_space.systemInput.shape == (1, 5)
    assert state_space.numberOfInputs == 1
    with pytest.raises(ValueError, match="Invalid observability method"):
        make_state_space(observabilityMethod="unknown")


def test_build_hankel_matrix_uses_column_major_signal_blocks():
    state_space = make_state_space()
    data = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])

    hankel = state_space.buildHankelMatrix(data)

    np.testing.assert_array_equal(
        hankel,
        [[1, 2, 3], [10, 20, 30], [2, 3, 4], [20, 30, 40]],
    )


def test_output_hankel_rejects_signals_with_different_lengths():
    outputs = np.empty(2, dtype=object)
    outputs[0] = np.array([1, 2, 3])
    outputs[1] = np.array([4, 5, 6, 7])
    state_space = make_state_space(system_output=outputs)

    with pytest.raises(ValueError, match="same length"):
        state_space.buildOutputHankelMatrix()


def test_cumulative_energy_truncation_respects_threshold_and_minimum_rank():
    state_space = make_state_space(energy_threshold=0.9)
    assert state_space.cumulativeEnergyTruncation(np.array([4.0, 2.0, 1.0])) == 2

    state_space.energy_threshold = 0.99
    assert state_space.cumulativeEnergyTruncation(np.array([4.0, 2.0, 1.0])) == 3

    state_space.energy_threshold = 0.1
    assert state_space.cumulativeEnergyTruncation(np.array([4.0, 2.0, 1.0])) == 2


def test_naishadham_factors_reconstruct_the_truncated_hankel_matrix():
    state_space = make_state_space(energy_threshold=0.8)
    hankel_output = np.array(
        [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.1]]
    )

    observability, state_matrix = state_space.buildObservability_Naishadham(
        hankel_output
    )

    assert observability.shape[1] == state_matrix.shape[0] == 2
    np.testing.assert_allclose(
        observability @ state_matrix,
        np.diag([3.0, 2.0, 0.0]),
        atol=1e-12,
    )


def test_rank_deficient_input_residual_covariance_is_finite_and_symmetric():
    state_space = make_state_space()
    input_hankel = np.ones((2, 5))
    output_hankel = np.array(
        [[1.0, 2.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0, 1.0]]
    )

    covariance = state_space.buildResidualCovarianceMatrix(
        input_hankel, output_hankel
    )

    assert np.all(np.isfinite(covariance))
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)


def test_weighted_observability_builds_multi_input_regression_blocks():
    inputs = np.array([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    state_space = make_state_space(
        system_input=inputs, system_output=np.zeros((1, inputs.shape[1]))
    )
    A = np.array([[0.5]])
    C = np.array([[2.0]])

    omega, weighted = state_space.buildWeightedObserability(A, C)

    np.testing.assert_allclose(omega[:, 0], [2.0, 1.0, 0.5])
    np.testing.assert_allclose(weighted[:, :2], inputs.T)
    np.testing.assert_allclose(
        weighted[:, 2:],
        [[0.0, 0.0], [4.0, 14.0], [8.0, 29.0]],
    )


def test_least_squares_recovers_input_parameters_and_initial_state():
    A = np.array([[0.6]])
    B = np.array([[1.25]])
    C = np.array([[1.5]])
    D = np.array([[0.2]])
    x0 = np.array([0.4])
    inputs = np.array([0.2, -0.7, 1.1, 0.3, -0.4, 0.8])
    _, outputs, _ = dlsim((A, B, C, D, 1.0), inputs, x0=x0)
    state_space = make_state_space(inputs, outputs.T)
    omega, weighted = state_space.buildWeightedObserability(A, C)

    estimated_B, estimated_D = state_space.build_B_D_matrices(
        A, C, omega, weighted, initialState=x0
    )
    estimated_x0 = state_space.estimate_initial_state(
        A, estimated_B, C, estimated_D, omega, weighted
    )

    np.testing.assert_allclose(estimated_B, B, atol=1e-12)
    np.testing.assert_allclose(estimated_D, D, atol=1e-12)
    np.testing.assert_allclose(estimated_x0, x0, atol=1e-12)
    assert state_space.compute_reconstruction_error(
        A, C, estimated_B, estimated_D, estimated_x0, omega, weighted
    ) == pytest.approx(0.0, abs=1e-12)


def test_evolve_input_accepts_time_major_and_input_major_arrays():
    A = np.array([[0.5]])
    B = np.array([[1.0, -0.25]])
    C = np.array([[2.0]])
    D = np.array([[0.1, 0.2]])
    inputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    state_space = make_state_space()

    states_time_major, outputs_time_major = state_space.evolveInput(
        A, B, C, D, inputs, np.array([0.0])
    )
    states_input_major, outputs_input_major = state_space.evolveInput(
        A, B, C, D, inputs.T, np.array([0.0])
    )

    np.testing.assert_allclose(states_input_major, states_time_major)
    np.testing.assert_allclose(outputs_input_major, outputs_time_major)
    assert outputs_time_major.shape == (1, inputs.shape[0])


@pytest.mark.parametrize("stabilizer", [stabilize_matrix, stabilize_schur_smooth])
def test_stabilizers_preserve_stable_matrices_and_stabilize_unstable_ones(
    stabilizer,
):
    stable = np.array([[0.5, 0.0], [0.0, -0.25]])
    unstable = np.array([[1.2, 0.0], [0.0, -1.5]])

    np.testing.assert_allclose(stabilizer(stable), stable, atol=1e-12)
    assert np.max(np.abs(np.linalg.eigvals(stabilizer(unstable)))) < 1.0
