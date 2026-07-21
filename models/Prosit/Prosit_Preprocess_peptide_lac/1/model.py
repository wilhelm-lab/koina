import triton_python_backend_utils as pb_utils
import numpy as np
from sequence_conversion import character_to_array, ALPHABET_MOD
import json
import re


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "peptides_in:0"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []

        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
            peptides_ = peptide_in.as_numpy().tolist()

            peptide_in_list = []
            for x in peptides_:
                seq = x[0].decode("utf-8")

                # Check if the sequence starts with []- or [UNIMOD:x]-
                if not re.match(r"^\[(?:UNIMOD:\d+)?\]-", seq):
                    seq = "[]-" + seq

                # Append -[] if it doesn't end with it
                if not seq.endswith("-[]"):
                    seq = seq + "-[]"

                peptide_in_list.append(seq)

            sequences = np.asarray(
                [character_to_array(seq).flatten() for seq in peptide_in_list]
            )

            t = pb_utils.Tensor("peptides_in:0", sequences.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
