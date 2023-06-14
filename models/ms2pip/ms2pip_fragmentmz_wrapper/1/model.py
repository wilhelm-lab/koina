import re
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from pyteomics import proforma


def internal_without_mods(sequences):
    """
    Function to remove any mod identifiers and return the plain AA sequence.
    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences
    """
    regex = r"\[.*?\]|\-"
    return [re.sub(regex, "", seq) for seq in sequences]


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None
        self.logger = pb_utils.Logger

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "mz")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            peptides = pb_utils.get_input_tensor_by_name(
                request, "peptides_in:0"
            ).as_numpy()

            fragmentmz = self.get_fragments(peptides)

            output_tensors = [
                pb_utils.Tensor("mz", fragmentmz.astype(self.output_dtype)),
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def get_fragments(self, peptides):
        tensor_inputs = [
            pb_utils.Tensor("ProForma", peptides.astype(np.object_).flatten()),
            pb_utils.Tensor("charges", np.array([1], dtype=np.int32)),
            pb_utils.Tensor("ion_series", np.array(["b", "y"], dtype=np.object_)),
        ]

        infer_request = pb_utils.InferenceRequest(
            model_name="fragment_mz",
            requested_output_names=["output_fragmentmz"],
            inputs=tensor_inputs,
        )

        resp = infer_request.exec()

        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        else:
            output = np.zeros((peptides.shape[0], 58), dtype=np.float128)

            tmp = pb_utils.get_output_tensor_by_name(
                resp, "output_fragmentmz"
            ).as_numpy()

            output[:, :29] = tmp[:, 0, 0, :29]  # b charge 1

            output[:, 29:] = np.flip(
                np.sort(tmp[:, 1, 0, :29]), 1
            )  # y charge 1 TODO clean up this mess

            output[output == 0] = -1

            return output
