#from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import requests

import yaml
import numpy as np
from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt



# ================== utils ==================

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def pre_process(smiles, precursor_type, collision_energy, config): 
    # 1. x_data: 3d molecules
    # Convert the SMILES string to a 3D molecular structure
    x_data = get_x_array(smiles, precursor_type, collision_energy, config)
    x_data = np.transpose(x_data, (1, 0))
        
    # 2. env_data: precursor type + normalised collision energy
    # Check if precursor_type is valid
    if precursor_type not in config['all']['precursor_type']:
        raise ValueError(f"Invalid precursor type: {precursor_type}")
    # One-hot encode the precursor type
    env_data = np.array(config['encoding']['precursor_type'][precursor_type])

    # Check if collision_energy is a number
    if not isinstance(collision_energy, (int, float)): 
        raise ValueError(f"Collision energy must be a number, got: {type(collision_energy).__name__}")
    # Convert collision_energy normalised collision energy
    charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}
    charge = int(config['encoding']['type2charge'][precursor_type])
    precursor_mz = precursor_calculator(precursor_type, mass=ExactMolWt(Chem.MolFromSmiles(smiles)))
    nce = collision_energy * 500 * charge_factor[charge] / precursor_mz

    # Concatenate precursor type and normalised collision energy
    env_data = np.concatenate([env_data, np.array([nce])], axis=0)
    
    # 3. idx_base_data: fixed value
    idx_base_data = np.array([[[0]]], dtype=np.int32)

    # This is for single data item now, let's expand it to batch size of 1
    x_data = np.expand_dims(x_data, axis=0).astype(np.float32)
    env_data = np.expand_dims(env_data, axis=0).astype(np.float32)
    idx_base_data = np.expand_dims(idx_base_data, axis=0)

    return x_data, env_data, idx_base_data


def get_x_array(smiles, precursor_type, collision_energy, config): 
    # Get xyz-coordinates and atom types of atoms in the molecule
    good_flg, xyz_arr, atom_types = conformation_array(smiles, config['encoding']['conf_type'])
    if not good_flg: 
        raise ValueError("Unsolvable SMILES: {}".format(smiles))
    
    # Check atom types and number of atoms
    valid_atom_types = set(config['all']['atom_type'])
    atom_types_set = set(atom_types)
    if not atom_types_set.issubset(valid_atom_types):
        raise ValueError(f"Invalid atom types in molecule: {atom_types_set - valid_atom_types}")
    
    atom_count = xyz_arr.shape[0]
    if not (config['all']['min_atom_num'] <= atom_count <= config['all']['max_atom_num']):
        raise ValueError(f"Number of atoms ({atom_count}) is out of range: "
              f"{config['all']['min_atom_num']} to {config['all']['max_atom_num']}")

    # Concatenate atom coordinates and atom types
    atom_type_one_hot = np.array([config['encoding']['atom_type'][atom] for atom in atom_types])
    assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]
    mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
    mol_arr = np.pad(mol_arr, ((0, config['encoding']['max_atom_num']-xyz_arr.shape[0]), (0, 0)), constant_values=0)

    return mol_arr

def conformation_array(smiles, conf_type): 
    # convert smiles to molecule
    if conf_type == 'etkdg': 
        mol = Chem.MolFromSmiles(smiles)
        mol_from_smiles = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_from_smiles)

    elif conf_type == 'etkdgv3': 
        mol = Chem.MolFromSmiles(smiles)
        mol_from_smiles = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 0xf00d
        AllChem.EmbedMolecule(mol_from_smiles, ps) 

    elif conf_type == '2d':
        mol = Chem.MolFromSmiles(smiles)
        mol_from_smiles = Chem.AddHs(mol)
        rdDepictor.Compute2DCoords(mol_from_smiles)

    else:
        raise ValueError('Unsupported conformation type. {}'.format(conf_type))

    # get the x,y,z-coordinates of atoms
    try: 
        conf = mol_from_smiles.GetConformer()
    except:
        return False, None, None
    xyz_arr = conf.GetPositions()

    # center the x,y,z-coordinates
    centroid = np.mean(xyz_arr, axis=0)
    xyz_arr -= centroid
    
    # concatenate with atom attributes
    xyz_arr = xyz_arr.tolist()
    for i, atom in enumerate(mol_from_smiles.GetAtoms()):
        xyz_arr[i] += [atom.GetDegree()]
        xyz_arr[i] += [atom.GetExplicitValence()]
        xyz_arr[i] += [atom.GetMass()/100]
        xyz_arr[i] += [atom.GetFormalCharge()]
        xyz_arr[i] += [atom.GetNumImplicitHs()]
        xyz_arr[i] += [int(atom.GetIsAromatic())]
        xyz_arr[i] += [int(atom.IsInRing())]
    xyz_arr = np.array(xyz_arr)
    
    # get the atom types of atoms
    atom_type = [atom.GetSymbol() for atom in mol_from_smiles.GetAtoms()]
    return True, xyz_arr, atom_type

def precursor_calculator(precursor_type, mass):
    if precursor_type == '[M+H]+':
        return mass + 1.007276 
    elif precursor_type == '[M+Na]+':
        return mass + 22.989218
    elif precursor_type == '[2M+H]+':
        return 2 * mass + 1.007276
    elif precursor_type == '[M-H]-':
        return mass - 1.007276
    elif precursor_type == '[M+H-H2O]+':
        return mass - 17.0038370665
    elif precursor_type == '[M+2H]2+':
        return mass/2 + 1.007276 
    else:
        raise ValueError('Unsupported precursor type: {}'.format(precursor_type))



# ================== test ==================

def test_interference_realmodel(): 
    bareserver = "localhost:8501"
    SERVER_HTTP = "http://localhost:8501"
    # bareserver = "ucr-lemon.duckdns.org:8501"
    # SERVER_HTTP = "http://ucr-lemon.duckdns.org:8501"
    MODEL_NAME = "3dmolms_torch"

    url = f"{SERVER_HTTP}/v2/models/{MODEL_NAME}/infer"

    triton_client = httpclient.InferenceServerClient(url=bareserver)

    # Check if the server is live
    if not triton_client.is_server_live():
        print("Triton server is not live!")
        exit(1)

    # Prepare input data
    smiles = 'C/C(=C\CNc1nc[nH]c2ncnc1-2)CO'
    precursor_type = '[M+H]+'
    collision_energy = 20
    config = load_config("./3dmolms_config.yml")
    x_data, env_data, idx_base_data = pre_process(smiles, precursor_type, collision_energy, config)
    
    # Create Triton inputs
    x_input = httpclient.InferInput("x", x_data.shape, "FP32")
    x_input.set_data_from_numpy(x_data)

    env_input = httpclient.InferInput("env", env_data.shape, "FP32")
    env_input.set_data_from_numpy(env_data)

    idx_base_input = httpclient.InferInput("idx_base", idx_base_data.shape, "INT32")
    idx_base_input.set_data_from_numpy(idx_base_data)

    # Define the output
    output = httpclient.InferRequestedOutput("3dmolms_out")

    # Perform inference
    response = triton_client.infer(MODEL_NAME, inputs=[x_input, env_input, idx_base_input], outputs=[output])

    # Get the output data
    output_data = response.as_numpy("3dmolms_out")
    print(output_data)
    print(output_data.shape)



def main():
    test_interference_realmodel()

if __name__ == "__main__":
    main()