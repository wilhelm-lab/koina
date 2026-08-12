import json
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "meta_input"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            charge = pb_utils.get_input_tensor_by_name(
                request, "precursor_charges_in:0"
            ).as_numpy()
            fragmentation = pb_utils.get_input_tensor_by_name(
                request, "fragmentation_types_encoding"
            ).as_numpy()
            mass = pb_utils.get_input_tensor_by_name(
                request, "precursor_mass"
            ).as_numpy()
            nce = pb_utils.get_input_tensor_by_name(
                request, "norm_collision_energy"
            ).as_numpy()

            meta = np.zeros((charge.shape[0], 3, 30))  # testing
            for i in range(charge.shape[0]):
                meta[i, 0, :] = charge[i, :]
                meta[i, 1, :] = fragmentation[i, :]
                meta[i, 2, 0] = mass[i]
                meta[i, 2, -1] = nce[i]

            t = pb_utils.Tensor("meta_input", meta.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
