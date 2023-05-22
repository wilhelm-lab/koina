import triton_python_backend_utils as pb_utils
import numpy as np
import json


def gen_annotation():
    ions = [
        "y",
        "b",
    ]
    charges = [1, 2, 3]
    positions = [x for x in range(1, 30)]
    annotation = []
    for pos in positions:
        for ion in ions:
            for charge in charges:
                annotation.append(f"{ion}{pos}+{charge}")
    return np.array(annotation).astype(np.object_)


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "annotation"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            batchsize = (
                pb_utils.get_input_tensor_by_name(request, "precursor_charge")
                .as_numpy()
                .shape[0]
            )
            annotation = np.tile(gen_annotation(), batchsize).reshape((-1, 174))
            t = pb_utils.Tensor("annotation", annotation)
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
