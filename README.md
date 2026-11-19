# Orthogonal Projection-Based State Space System Identification (OPB-SSSI)

## Quick start

### System identification

Load the experimental data (`time_input`, `input_signal`, `time_output_raw`,
`output_signal_raw`) and align the signals on a common sampling vector:

```python
import numpy as np
from src.system_identification_wrapper import SystemIdentificationWrapper
from src.system_identification import StateSpace
from src.evt_statistical_bound import EVTStatisticalBound

training_time_output = np.arange(start, stop, step)
system = SystemIdentificationWrapper(timeInput=time_input, timeOutput=training_time_output)
system.addInputData(input_signal)
system.buildInterpolatedInputValues()
system.addOutputData(np.interp(training_time_output, time_output_raw, output_signal_raw))

state_space = StateSpace(systemInput=system.interpolatedInputValues[0],
                         systemOutput=system.outputValues)
A, B, C, D, initial_state = state_space.buildStateSpaceSystem()
```

The algorithm returns the matrices of the discrete-time model
`x[k+1] = A x[k] + B u[k]`, `y[k] = C x[k] + D u[k]`, and the initial state.
For `n` states, `m` inputs, and `p` outputs, their general dimensions are
`A: (n, n)`, `B: (n, m)`, `C: (p, n)`, and `D: (p, m)`.

### System evolution

The identified matrices can also evolve the model with an input sequence
covering a longer time range than the training data. Use the same sampling
interval and align the input samples with `prediction_time`:

```python
prediction_time = np.arange(start, prediction_stop, step)
# input_prediction contains the input values at prediction_time
x_prediction, y_prediction = state_space.evolveInput(
    A, B, C, D,
    u=input_prediction,
    x0=initial_state,
)
```

`y_prediction` is the estimated output at each sample in `prediction_time`;
`x_prediction` contains the corresponding state evolution.

### EVT statistical bound

Reconstruct the output to obtain residuals, then estimate `Gamma` for the
requested horizon and confidence level:

```python
_, y_est = state_space.evolveInput(A, B, C, D,
                                   u=system.interpolatedInputValues[0],
                                   x0=initial_state)
residuals = system.outputValues[0] - y_est[0]
evt = EVTStatisticalBound(residuals, number_of_pending_iterations=1000)
Gamma = evt.compute_statistical_bound()
info = evt.get_statistical_info()
```

`info` contains the POT threshold and GPD fit statistics. To get independent
bounds for multiple outputs, apply EVT to each output's residuals separately.
