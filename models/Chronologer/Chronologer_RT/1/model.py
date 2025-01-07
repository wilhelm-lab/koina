import re
import json
import numpy as np
import triton_python_backend_utils as pb_utils


def internal_without_mods(sequences):
    """
    Function to remove any mod identifiers and return the plain AA sequence.
    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences
    """
    regex = r"\[.*?\]|\-"
    return [re.sub(regex, "", seq.decode("utf-8")) for seq in sequences]


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None

    def initialize(self, args):
        model_config = model_config = json.loads(args["model_config"])
        out_config = pb_utils.get_output_config_by_name(model_config, "irt")
        self.output_dtype = pb_utils.triton_string_to_numpy(out_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            peptide_in = (
                pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
                .as_numpy()
                .flatten()
            )

            peptide_in_nomod = internal_without_mods(peptide_in)
            pep_len = np.vectorize(len)(peptide_in_nomod)

            oirt = np.full((peptide_in.shape[0], 1), -1, dtype=np.float32)
            for l in set(pep_len):
                idx = l == pep_len
                tmp = self.predict_batch(peptide_in[idx]).flatten()
                oirt[idx, 0] = tmp

            output_tensors = [pb_utils.Tensor("irt", oirt.astype(self.output_dtype))]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def predict_batch(self, seq):
        tensor_inputs = [
            pb_utils.Tensor("peptide_sequences", seq.reshape((-1, 1))),
        ]

        infer_request = pb_utils.InferenceRequest(
            model_name="Chronologer_RT_sb",
            requested_output_names=["irt"],
            inputs=tensor_inputs,
        )

        resp = infer_request.exec()

        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        else:
            return pb_utils.get_output_tensor_by_name(resp, "irt").as_numpy()
