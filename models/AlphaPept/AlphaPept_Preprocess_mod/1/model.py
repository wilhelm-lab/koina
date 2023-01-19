import triton_python_backend_utils as pb_utils
import numpy as np
import pandas as pd
import json

def _parse_mod_formula(formula, mod_elem_to_idx, mod_feature_size):
    '''
    Parse a modification formula to a feature vector
    '''
    feature = np.zeros(mod_feature_size)
    elems = formula.strip(')').split(')')
    for elem in elems:
        chem, num = elem.split('(')
        num  = int(num)
        if chem in mod_elem_to_idx:
            feature[mod_elem_to_idx[chem]] = num
        else:
            feature[-1] += num
    return feature


def convert(mods, mod_sites, nAA):

    MOD_DF = pd.read_csv("/models/AlphaPept/AlphaPept_Preprocess_mod/1/mod_df.csv")

    mod_elements = ['C', 'H', 'N', 'O', 'P', 'S', 'B', 'F', 'I', 'K', 'U', 'V', 'W',
                'X', 'Y', 'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'Ba',
                'Be', 'Bi', 'Bk', 'Br', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co',
                'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Es', 'Eu', 'Fe', 'Fm', 'Fr', 'Ga',
                'Gd', 'Ge', 'He', 'Hf', 'Hg', 'Ho', 'In', 'Ir', 'Kr', 'La', 'Li',
                'Lr', 'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Nd', 'Ne', 'Ni',
                'No', 'Np', 'Os', 'Pa', 'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu',
                'Ra', 'Rb', 'Re', 'Rh', 'Rn', 'Ru', 'Sb', 'Sc', 'Se', 'Si', 'Sm',
                'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'Xe',
                'Yb', 'Zn', 'Zr', '2H', '13C', '15N', '18O', '?']
    mod_feature_size = len(mod_elements)
    mod_elem_to_idx = dict(zip(mod_elements, range(mod_feature_size)))

    MOD_TO_FEATURE = {}
    def update_all_mod_features():
        for modname, formula in MOD_DF[['mod_name','composition']].values:
            MOD_TO_FEATURE[modname] = _parse_mod_formula(formula, mod_elem_to_idx, mod_feature_size)
    update_all_mod_features()


    mod_features_list = pd.Series(mods).str.split(';').apply(
        lambda mod_names: [
            MOD_TO_FEATURE[mod] for mod in mod_names
            if len(mod) > 0
        ]
    )
    mod_sites_list = pd.Series(mod_sites).str.split(';').apply(
        lambda mod_sites: [
            int(site) for site in mod_sites
            if len(site) > 0
        ]
    )
    mod_x_batch = np.zeros(
        (len(nAA.as_numpy()), int(nAA.as_numpy()[0])+2, mod_feature_size)
    )
    for i, (mod_feats, mod_sites) in enumerate(
        zip(mod_features_list, mod_sites_list)
    ):
        if len(mod_sites) > 0:
            for site, feat in zip(mod_sites, mod_feats):
                # Process multiple mods on one site
                mod_x_batch[i, site, :] += feat
            # mod_x_batch[i,mod_sites,:] = mod_feats
    return mod_x_batch

class TritonPythonModel:
   def initialize(self,args):
      self.model_config = model_config = json.loads(args['model_config'])
      output0_config = pb_utils.get_output_config_by_name(
              self.model_config, "output:0")
      self.output_dtype = pb_utils.triton_string_to_numpy(
                          output0_config['data_type'])

   def execute(self, requests):
     peptide_in_str = []
     responses = []
     for request in requests:
       mods = pb_utils.get_input_tensor_by_name(request, "mods:0")
       mod_sites = pb_utils.get_input_tensor_by_name(request, "mod_sites:0")
       nAA = pb_utils.get_input_tensor_by_name(request, "nAA:0")
      
       mods_ = mods.as_numpy().tolist()
       mods_list = [x[0].decode('utf-8')  for x in mods_ ]

       mod_sites_ = mod_sites.as_numpy().tolist()
       mod_sites_list = [x[0].decode('utf-8')  for x in mod_sites_ ]

       output = convert(mods_list, mod_sites_list, nAA)
       t = pb_utils.Tensor("output:0",output.astype(self.output_dtype) )
       responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
     return responses
   
   def finalize(self):
     print('done processing Preprocess')
