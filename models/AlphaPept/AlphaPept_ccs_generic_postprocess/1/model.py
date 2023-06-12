import re
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None
        self.logger = pb_utils.Logger

    def initialize(self, args):
        print("Preprocessing of the Peptide_input")
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "ccs")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            ccs = pb_utils.get_input_tensor_by_name(request, "ccs_raw").as_numpy()

            output_tensors = [pb_utils.Tensor("ccs", ccs.astype(self.output_dtype))]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses
