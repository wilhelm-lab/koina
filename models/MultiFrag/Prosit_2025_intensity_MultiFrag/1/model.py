import numpy as np
import pandas as pd
import re
import json
import triton_python_backend_utils as pb_utils
import os
from MultiFrag.Prosit_2025_intensity_MultiFrag.mass_scale import Scale

def tokenize_modified_sequence(modseq):
    tokenized = []
    modseq = re.sub('-|(\[])', '', modseq) # remove - or []

    pos = 0
    while pos < len(modseq):
        character = modseq[pos]
        hx = ord(character)
        if character == '[':
            ahead = 1
            mod = []
            while character != ']':
                mod.append(character)
                character = modseq[pos+ahead]
                ahead += 1
            token = "".join(mod) + ']'
            if pos != 0:
                tokenized[-1] += token
            else:
                tokenized.append(token)
            pos += ahead - 1
        else:
            tokenized.append(character)
        pos += 1

    return tokenized

def create_dictionary(dictionary_path):
    amod_dic = {
        line.split()[0]:m for m, line in enumerate(open(dictionary_path))
    }
    amod_dic['X'] = len(amod_dic)
    amod_dic[''] = amod_dic['X']
    return amod_dic

class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None
        self.logger = pb_utils.Logger

        P = "MultiFrag/Prosit_2025_intensity_MultiFrag"

        self.ion_dict = pd.read_csv(os.path.join(P, "filtered_ion_dict.csv"), index_col='full')
        num_tokens = len(open(os.path.join(P, "token_dictionary.txt")).read().strip().split("\n")) + 1
        final_units = len(self.ion_dict)
        max_charge = 6
                
        self.token_dict = create_dictionary(os.path.join(P, "token_dictionary.txt"))
        self.token_dict['C'] = self.token_dict['C[UNIMOD:4]']
        self.max_length = 30
        
        method_list = ['ECD', 'EID', 'UVPD', 'HCD', 'ETciD']
        self.method_dic = {method: m for m, method in enumerate(method_list)}
        self.method_dicr = {n:m for m,n in self.method_dic.items()}

        self.scale = Scale()

    def filter_fake(self, peptide_length, precursor_charge):
        return np.array(
            (self.ion_dict['length'] >= peptide_length) |
            (self.ion_dict['charge'] > precursor_charge)
        )

    def batch_mz(self, peptides, lengths, charges):
        mz = np.zeros((len(peptides), 815)).astype(np.float32)
        for m, (seq, length, charge) in enumerate(zip(peptides, lengths, charges)):
            mzs = np.array(
                [self.scale.calcmass(seq, charge, ion) for ion in self.ion_dict.index]
            )
            mask = self.filter_fake(length, charge)
            mzs[mask] = -1
            mz[m] = mzs

        return mz

    def execute(self, requests):
        responses = []
        for request in requests:
            
            peptide_in = (
                pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
                .as_numpy()
                .flatten()
            ).astype('U200') # binary (b'word') to string ('word')
            charge_in = (
                pb_utils.get_input_tensor_by_name(request, "precursor_charges")
                .as_numpy()
                .flatten()
            )
            method_in = (
                pb_utils.get_input_tensor_by_name(request, "fragmentation_types")
                .as_numpy()
                .flatten()
            ).astype('U46')
            
            # Process the peptide sequences
            ts = [[self.token_dict[y] for y in tokenize_modified_sequence(x)] for x in peptide_in]
            # Get the peptide lengths
            peptide_lengths = np.array([len(m) for m in ts])
            n_term_mask = [True if '[UNIMOD:1]' in seq else False for seq in peptide_in]
            peptide_lengths[n_term_mask] -= 1
            # Pad peptide sequences
            tokenized_sequence = np.stack(
                x + (self.max_length-len(x))*[self.token_dict['X']]
                for x in ts
            )
            # Turn method strings into method integers
            method_vector = np.array([self.method_dic[x] for x in method_in])
            
            # Ensure correct shapes and data types
            tokenized_sequence = tokenized_sequence.astype(np.int32)
            charge_in_ = charge_in[:, None].astype(np.int32)
            method_vector = method_vector[:,None].astype(np.int32)
            
            # Call the base model
            ints = self.predict_batch(tokenized_sequence, charge_in_, method_vector)
            ints = np.clip(ints[0] / ints[0].max(1, keepdims=True), 0, 1)

            # Create the other outputs
            mzs = self.batch_mz(peptide_in, peptide_lengths, charge_in)
            anns = np.tile(self.ion_dict.index.to_numpy()[None], [len(charge_in), 1])

            output_tensors = [
                pb_utils.Tensor("intensities", ints.astype(self.output_dtype)),
                pb_utils.Tensor("mz", mzs.astype(self.output_dtype)),
                pb_utils.Tensor("annotation", anns.astype(np.object_)),
            ]
            
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def predict_batch(self, intseq, charge, method):
        dummy_energy = np.ones( ( len(charge), 1 ) , dtype=np.float32)
        assert charge.shape == dummy_energy.shape == method.shape, charge.shape
        tensor_inputs = [
            pb_utils.Tensor("intseq", intseq),
            pb_utils.Tensor("charge", charge),
            pb_utils.Tensor("energy", dummy_energy),
            pb_utils.Tensor("method", method),
        ]

        infer_request = pb_utils.InferenceRequest(
            model_name="multifrag25",
            requested_output_names=["intensities"],
            inputs=tensor_inputs,
            preferred_memory=pb_utils.PreferredMemory(
                pb_utils.TRITONSERVER_MEMORY_CPU, 0
            ),
        )

        resp = infer_request.exec()

        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        else:
            output = [
                pb_utils.get_output_tensor_by_name(resp, "intensities").as_numpy(),
            ]

            return output
