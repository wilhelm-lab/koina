import triton_python_backend_utils as pb_utils
import numpy as np
from sequence_conversion import character_to_array, ALPHABET_MOD
import json
import re


def internal_without_mods(sequences):
    """
    Function to remove any mod identifiers and return the plain AA sequence.
    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences
    """
    regex = r"\[.*?\]|\-"
    return [re.sub(regex, "", seq) for seq in sequences]


class TritonPythonModel:
    def initialize(self, args):
        print("Preprocessing of the Peptide_input")
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "peptides_in:0"
        )
        print("preprocess_peptide type: " + str(output0_config))
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        peptide_in_str = []
        responses = []
        logger = pb_utils.Logger
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
            peptides_ = peptide_in.as_numpy().tolist()
            peptide_in_list = [x[0].decode("utf-8") for x in peptides_]
            logger.log_info(str(peptide_in_list))

            sequences = np.asarray(
                [character_to_array(seq).flatten() for seq in peptide_in_list]
            )
            logger.log_info(str(sequences))

            t = pb_utils.Tensor("peptides_in:0", sequences.astype(self.output_dtype))

            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
            print("sequences: ")
            print(len(sequences))
            print(sequences)
        return responses

    def finalize(self):
        print("done processing Preprocess")
