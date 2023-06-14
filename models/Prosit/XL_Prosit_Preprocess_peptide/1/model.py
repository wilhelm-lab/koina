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
        output0_config_1 = pb_utils.get_output_config_by_name(
            self.model_config, "peptides_in_1:0"
        )
        output0_config_2 = pb_utils.get_output_config_by_name(
            self.model_config, "peptides_in_2:0"
        )
        print("preprocess_peptide type: " + str(output0_config_1))
        print("preprocess_peptide type: " + str(output0_config_2))
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config_1["data_type"])
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config_2["data_type"])

    def execute(self, requests):
        peptide_in_str = []
        responses = []
        logger = pb_utils.Logger
        for request in requests:
            peptide_in_1 = pb_utils.get_input_tensor_by_name(request, "peptide_sequences_1")
            peptide_in_2 = pb_utils.get_input_tensor_by_name(request, "peptide_sequences_2")
            peptides_1 = peptide_in_1.as_numpy().tolist()
            peptides_2 = peptide_in_2.as_numpy().tolist()
            peptide_in_1_list = [x[0].decode("utf-8") for x in peptides_1]
            peptide_in_2_list = [x[0].decode("utf-8") for x in peptides_2]
            logger.log_info(str(peptide_in_1_list))
            logger.log_info(str(peptide_in_2_list))

            sequences_1 = np.asarray(
                [character_to_array(seq).flatten() for seq in peptide_in_1_list]
            )
            sequences_2 = np.asarray(
                [character_to_array(seq).flatten() for seq in peptide_in_2_list]
            )
            logger.log_info(str(sequences_1))
            logger.log_info(str(sequences_2))

            t_1 = pb_utils.Tensor("peptides_in_1:0", sequences_1.astype(self.output_dtype))
            t_2 = pb_utils.Tensor("peptides_in_2:0", sequences_2.astype(self.output_dtype))

            responses.append(pb_utils.InferenceResponse(output_tensors=[t_1]))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t_2]))
            print("sequences_1: ")
            print("sequences_2: ")
            print(len(sequences_1))
            print(len(sequences_2))
            print(sequences_1)
            print(sequences_2)
        return responses

    def finalize(self):
        print("done processing Preprocess")