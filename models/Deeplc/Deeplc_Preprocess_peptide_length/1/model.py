import triton_python_backend_utils as pb_utils
import numpy as np
import json
import re


class TritonPythonModel:
    def initialize(self, args):
        print("Preprocessing of the Peptide_input")
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "peptide_length"
        )
        print("preprocess_peptide type: " + str(output0_config))
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        peptide_in_str = []
        responses = []
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "stripped_peptide")
            peptides_ = peptide_in.as_numpy().tolist()
            peptide_in_list = [x[0].decode("utf-8") for x in peptides_]

            peptide_lengths = np.asarray([[len(pep)] for pep in peptide_in_list])
            print(peptide_lengths)
            t = pb_utils.Tensor(
                "peptide_length", peptide_lengths.astype(self.output_dtype)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
            print("sequences: ")
            print(len(peptide_lengths))
            print(peptide_lengths)
        return responses

    def finalize(self):
        print("done processing Preprocess")
