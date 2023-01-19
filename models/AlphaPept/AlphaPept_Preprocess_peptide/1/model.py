import triton_python_backend_utils as pb_utils
import numpy as np
import json

def character_to_array(peptide_in_list):
    seq_array = np.array(peptide_in_list).astype('U')
    x = np.array(seq_array).view(np.int32).reshape(
                len(seq_array), 
                    -1) - ord('A')+1
    return np.pad(x, [(0,0)]*(len(x.shape)-1)+[(1,1)])


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
      sequences = character_to_array(peptide_in_list)
      t = pb_utils.Tensor("peptides_in:0",sequences.astype(self.output_dtype) )
      responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
      print("sequences: ")
      print(len(sequences))
      print(sequences)
     return responses
   def finalize(self):
     print('done processing Preprocess')
