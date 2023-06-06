import triton_python_backend_utils as pb_utils
import numpy as np
import json

MAX_CHARGE = 6


def indices_to_one_hot(data, nb_classes):
    """
    Convert an iterable of indices to one-hot encoded labels.
    :param data: charge, int between 1 and 6
    """
    targets = np.array([data - 1])  # -1 for 0 indexing
    return np.int_((np.eye(nb_classes)[targets])).tolist()[0]


def to_on_hot(numeric):
    array = [indices_to_one_hot(x, MAX_CHARGE) for x in numeric]
    return np.array(array, dtype=float)


class TritonPythonModel:
    def initialize(self, args):
        print("Preprocessing of the precursor_charges_In")
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "precursor_charges_in:0"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        peptide_in_str = []
        responses = []
        print("Pre-processing of charge is called")
        for request in requests:
            charge_in_raw = pb_utils.get_input_tensor_by_name(
                request, "precursor_charges"
            )
            charge_in_flat = sum(charge_in_raw.as_numpy().tolist(), [])
            charge_in = to_on_hot(charge_in_flat)
            t = pb_utils.Tensor(
                "precursor_charges_in:0", charge_in.astype(self.output_dtype)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
            print("charge_in: ")
            print(len(charge_in))
            print(charge_in)
        return responses

    def finalize(self):
        print("done processing Preprocess charge")
