import json
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = []

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "out_model1")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "in_model1")
            import sys
            print("XXXXXXXXXXXXXXXXX", raw, file=sys.stderr, flush=True)
            norm = raw.as_numpy() * 2
            ce_tensor = pb_utils.Tensor("out_model1", norm.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[ce_tensor]))

        return responses

    def finalize(self):
        pass