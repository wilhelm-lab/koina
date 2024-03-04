import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "instrument_types_encoding"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            instrument_types = np.char.lower(pb_utils.get_input_tensor_by_name(
                request, "instrument_types"
            ).as_numpy().astype(str))

            instrument_types_encoding = np.full(instrument_types.shape, -1)
            for k, v in {"qe": 0, "lumos": 1, "timstof": 2, "sciextof": 3}.items():
                instrument_types_encoding[instrument_types == k] = v

            t = pb_utils.Tensor(
                "instrument_types_encoding",
                instrument_types_encoding.astype(self.output_dtype),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
