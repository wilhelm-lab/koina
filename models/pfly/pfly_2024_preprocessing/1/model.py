import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def initialize(self, args):
        # Define amino acid encoder
        self.aa_encoder = {
            "0": 0,
            "A": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "K": 9,
            "L": 10,
            "M": 11,
            "N": 12,
            "P": 13,
            "Q": 14,
            "R": 15,
            "S": 16,
            "T": 17,
            "V": 18,
            "W": 19,
            "Y": 20,
        }

    def execute(self, requests):
        responses = []

        for request in requests:
            # Retrieve input tensor
            peptide_input = pb_utils.get_input_tensor_by_name(
                request, "peptide_sequences"
            ).as_numpy()

            # List to hold encoded sequences (shape: [batch_size, 40])
            batch_encoded_sequences = []

            for seq in peptide_input:
                seq = seq[0].decode("utf-8")
                seq = seq + "0" * (40 - len(seq))

                # Encode sequence
                encoded_seq = [self.aa_encoder[aa] for aa in seq]

                # Append the sequence to the batch (each encoded sequence is a list of 40 integers)
                batch_encoded_sequences.append(encoded_seq)

            # Convert batch to NumPy array with dtype=int64 and shape [batch_size, 40]
            # This ensures the final shape is [batch_size, 40]
            batch_encoded_tensor = pb_utils.Tensor(
                "encoded_sequences", np.array(batch_encoded_sequences, dtype=np.int64)
            )

            # Add response
            responses.append(pb_utils.InferenceResponse([batch_encoded_tensor]))

        return responses

    def finalize(self):
        pass
