import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        super().__init__()


    def execute(self, requests):
        responses = []
        for request in requests:

            charges_in = (
                pb_utils.get_input_tensor_by_name(request, "precursor_charges")
                .as_numpy()
                .astype(np.float32)
            )

            t = pb_utils.Tensor("precursor_charges_FP", charges_in)
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))

        return responses

    def finalize(self):
        pass
