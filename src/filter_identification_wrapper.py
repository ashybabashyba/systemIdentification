import numpy as np

class FilterIdentificationWrapper:
    def __init__(self):
        self.inputValues = []

        self.referenceOutputValues = []
        self.interpolatedReferenceOutputValues = []

        self.deterministicOutputValues = []
        self.interpolatedDeterministicOutputValues = []

        self.inputTimeData  = []
        self.deterministicOutputTimeData = []
        self.referenceOutputTimeData = []
        
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

        return self.interpolatedReferenceOutputValues - self.interpolatedDeterministicOutputValues