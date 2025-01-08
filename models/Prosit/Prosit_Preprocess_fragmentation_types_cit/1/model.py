import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None
        self.logger = pb_utils.Logger

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "fragmentation_types_encoding"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            fragmentation_types = pb_utils.get_input_tensor_by_name(
                request, "fragmentation_types"
            ).as_numpy()

            fragmentation_types_encoding = np.zeros(fragmentation_types.shape)
            for k, v in {"HCD": 1, "CID": 2}.items():
                fragmentation_types_encoding[fragmentation_types == str.encode(k)] = v

            t = pb_utils.Tensor(
                "fragmentation_types_encoding",
                fragmentation_types_encoding.astype(self.output_dtype),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
