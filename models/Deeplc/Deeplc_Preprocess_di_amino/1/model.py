import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "diamino_ac"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        peptide_in_str = []
        responses = []
        for request in requests:
            ac_in = pb_utils.get_input_tensor_by_name(request, "single_ac")
            single_ac = ac_in.as_numpy()

            fill = np.add.reduceat(single_ac, range(0, 60, 2), axis=1)
            t = pb_utils.Tensor("diamino_ac", fill.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass