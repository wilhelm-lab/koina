import json
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None
        self.logger = pb_utils.Logger

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "irt_norm")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            irt_raw = pb_utils.get_input_tensor_by_name(
                request, "in/irt_raw"
            ).as_numpy()

            irt_norm = irt_raw * 43.39373 + 56.35363441

            output_tensors = [
                pb_utils.Tensor("irt_norm", irt_norm.astype(self.output_dtype))
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses
