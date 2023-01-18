import triton_python_backend_utils as pb_utils
import numpy as np
import json
from postprocess import create_masking, apply_masking

class TritonPythonModel:
   def initialize(self,args):
      print("Preprocessing of the Peptide_input")
      self.model_config = model_config = json.loads(args['model_config'])
      output0_config = pb_utils.get_output_config_by_name(
              self.model_config, "out/Reshape:1")
      self.output_dtype = pb_utils.triton_string_to_numpy(
                          output0_config['data_type'])

   def execute(self, requests):
      peptide_in_str = []
      responses = []
      for request in requests:
        peptide_in = pb_utils.get_input_tensor_by_name(request, "peptides_in:0").as_numpy().tolist()
        precursor_charge_in = pb_utils.get_input_tensor_by_name(request, "precursor_charge_in:0").as_numpy().tolist()
        peaks_in = pb_utils.get_input_tensor_by_name(request, "peaks_in:0").as_numpy().tolist()
        peptide_lengths = []
        for batch in peptide_in:
          for peptide in batch:
            peptide_lengths.append(len(peptide))
        mask = create_masking(precursor_charge_in,peptide_lengths)
        peaks = apply_masking(peaks_in,mask)
        peaks = np.array(peaks,dtype=float)
        t = pb_utils.Tensor("out/Reshape:1",peaks.astype(self.output_dtype))
        responses.append(pb_utils.InferenceResponse(output_tensors=[t]))

      return responses
   def finalize(self):
     print('done processing Preprocess')