import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "collision_energy_in:0"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            charge_in_raw = pb_utils.get_input_tensor_by_name(
                request, "precursor_charge_in_int:0"
            ).as_numpy()

            charge_in_raw = charge_in_raw.astype(self.output_dtype)
            charge_in_raw.fill(0.35)
            t = pb_utils.Tensor(
                "collision_energy_in:0", charge_in_raw
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
