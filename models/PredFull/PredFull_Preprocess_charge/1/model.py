import json
import triton_python_backend_utils as pb_utils
import numpy as np

MAX_CHARGE = 30


def indices_to_one_hot(data, nb_classes):
    """
    Convert an iterable of indices to one-hot encoded labels.
    :param data: charge, int between 1 and 30
    """
    if data > nb_classes or data < 1:
        raise RuntimeError("Charge out of range")
    targets = np.array([data - 1])  # -1 for 0 indexing
    return np.int_((np.eye(nb_classes)[targets])).tolist()[0]


def to_one_hot(numeric):
    array = [indices_to_one_hot(x, MAX_CHARGE) for x in numeric]
    return np.array(array, dtype=float)


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "precursor_charges_in:0"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):  # throw error if charge less than 1 or greater than 30
        responses = []
        for request in requests:
            # at earliest sign of charge exceeding max, can append TritonError instead
            charge_in_raw = pb_utils.get_input_tensor_by_name(
                request, "precursor_charges"
            )
            charge_in_flat = sum(charge_in_raw.as_numpy().tolist(), [])
            charge_in = to_one_hot(charge_in_flat)
            t = pb_utils.Tensor(
                "precursor_charges_in:0", charge_in.astype(self.output_dtype)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
