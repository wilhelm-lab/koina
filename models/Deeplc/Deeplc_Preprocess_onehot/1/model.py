import triton_python_backend_utils as pb_utils
import numpy as np
from sequence_conversion import character_to_array, ALPHABET_MOD
import json

def indices_to_one_hot(data, nb_classes):
    """
    Convert an iterable of indices to one-hot encoded labels.
    :param data: charge, int between 1 and 6
    """
    targets = np.array([data])  # -1 for 0 indexing
    return np.int_((np.eye(nb_classes)[targets])).tolist()[0]
    
def one_hot_encoding(unmod_sequences):
    numeric =[dict_aa[x] for unmod_seq in unmod_sequences for x in unmod_seq]
    array = [indices_to_one_hot(x, 20) for x in numeric]
    return np.array(array,dtype=float)
    
dict_aa={
     "K": 0,
     "R": 1,
     "P": 2,
     "T": 3,
     "N": 4,
     "A": 5,
     "Q": 6,
     "V": 7,
     "S": 8,
     "G": 9,
     "I": 10,
     "L": 11,
     "C": 12,
     "M": 13,
     "H": 14,
     "F": 15,
     "Y": 16,
     "W": 17,
     "E": 18,
     "D": 19
}

class TritonPythonModel:
   def initialize(self,args):
      print("Preprocessing of the Peptide_input")
      self.model_config = model_config = json.loads(args['model_config'])
      output0_config = pb_utils.get_output_config_by_name(
              self.model_config, "peptides_in:0")
      print("preprocess_peptide type: " + str(output0_config))
      self.output_dtype = pb_utils.triton_string_to_numpy(
                          output0_config['data_type'])
   def execute(self, requests):
     peptide_in_str = []
     responses = []
     for request in requests:
      peptide_in = pb_utils.get_input_tensor_by_name(request, "peptides_in_str:0")
      peptides_ = peptide_in.as_numpy().tolist()
      peptide_in_list = [x[0].decode('utf-8')  for x in peptides_ ]

      fill = one_hot_encoding(peptide_in_list)
      sequences = np.zeros([60,20])
      sequences[:,:fill.shape[0],] = fill
      t = pb_utils.Tensor("peptides_in:0",sequences.astype(self.output_dtype) )
      responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
      print("sequences: ")
      print(len(sequences))
      print(sequences)
     return responses
   def finalize(self):
     print('done processing Preprocess')
