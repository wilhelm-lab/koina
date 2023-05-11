import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "norm_collision_energy"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            raw_ce = pb_utils.get_input_tensor_by_name(
                request, "raw_collision_energy"
            ).as_numpy()

            t = pb_utils.Tensor(
                "norm_collision_energy", (raw_ce / 100).astype(self.output_dtype)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
