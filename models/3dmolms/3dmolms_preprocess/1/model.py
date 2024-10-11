import json
import triton_python_backend_utils as pb_utils
import os
import numpy as np
from rdkit import Chem

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt


def load_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def pre_process(smiles, precursor_type, collision_energy, config):
    # 1. x_data: 3d molecules
    # Convert the SMILES string to a 3D molecular structure
    x_data = get_x_array(smiles, precursor_type, collision_energy, config)
    x_data = np.transpose(x_data, (1, 0))
    print("done converting smiles to a 3d molecular structure")

    # 2. env_data: precursor type + normalised collision energy
    # Check if precursor_type is valid
    if precursor_type not in config["all"]["precursor_type"]:
        raise ValueError(f"Invalid precursor type: {precursor_type}")
    # One-hot encode the precursor type
    env_data = np.array(config["encoding"]["precursor_type"][precursor_type])

    # Check if collision_energy is a number
    if not isinstance(collision_energy, (int, float)):
        raise ValueError(
            f"Collision energy must be a number, got: {type(collision_energy).__name__}"
        )
    # Convert collision_energy normalised collision energy
    charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}
    charge = int(config["encoding"]["type2charge"][precursor_type])
    precursor_mz = precursor_calculator(
        precursor_type, mass=ExactMolWt(Chem.MolFromSmiles(smiles))
    )
    nce = collision_energy * 500 * charge_factor[charge] / precursor_mz

    # Concatenate precursor type and normalised collision energy
    env_data = np.concatenate([np.array([nce]), env_data], axis=0)

    # 3. idx_base_data: fixed value
    idx_base_data = np.array([[[0]]], dtype=np.int32)

    # This is for single data item now, let's expand it to batch size of 1
    x_data = np.expand_dims(x_data, axis=0).astype(np.float32)
    env_data = np.expand_dims(env_data, axis=0).astype(np.float32)
    idx_base_data = np.expand_dims(idx_base_data, axis=0)

    return x_data, env_data, idx_base_data


def get_x_array(smiles, precursor_type, collision_energy, config):
    # Get xyz-coordinates and atom types of atoms in the molecule
    good_flg, xyz_arr, atom_types = conformation_array(
        smiles, config["encoding"]["conf_type"]
    )
    if not good_flg:
        raise ValueError("Unsolvable SMILES: {}".format(smiles))

    # Check atom types and number of atoms
    valid_atom_types = set(config["all"]["atom_type"])
    atom_types_set = set(atom_types)
    if not atom_types_set.issubset(valid_atom_types):
        raise ValueError(
            f"Invalid atom types in molecule: {atom_types_set - valid_atom_types}"
        )

    atom_count = xyz_arr.shape[0]
    if not (
        config["all"]["min_atom_num"] <= atom_count <= config["all"]["max_atom_num"]
    ):
        raise ValueError(
            f"Number of atoms ({atom_count}) is out of range: "
            f"{config['all']['min_atom_num']} to {config['all']['max_atom_num']}"
        )

    # Concatenate atom coordinates and atom types
    atom_type_one_hot = np.array(
        [config["encoding"]["atom_type"][atom] for atom in atom_types]
    )
    assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]
    mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
    mol_arr = np.pad(
        mol_arr,
        ((0, config["encoding"]["max_atom_num"] - xyz_arr.shape[0]), (0, 0)),
        constant_values=0,
    )

    return mol_arr


def conformation_array(smiles, conf_type):
    # convert smiles to molecule
    if conf_type == "etkdg":
        mol = Chem.MolFromSmiles(smiles)
        mol_from_smiles = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_from_smiles)

    elif conf_type == "etkdgv3":
        mol = Chem.MolFromSmiles(smiles)
        mol_from_smiles = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 0xF00D
        AllChem.EmbedMolecule(mol_from_smiles, ps)

    elif conf_type == "2d":
        mol = Chem.MolFromSmiles(smiles)
        mol_from_smiles = Chem.AddHs(mol)
        rdDepictor.Compute2DCoords(mol_from_smiles)

    else:
        raise ValueError("Unsupported conformation type. {}".format(conf_type))

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
        xyz_arr[i] += [atom.GetMass() / 100]
        xyz_arr[i] += [atom.GetFormalCharge()]
        xyz_arr[i] += [atom.GetNumImplicitHs()]
        xyz_arr[i] += [int(atom.GetIsAromatic())]
        xyz_arr[i] += [int(atom.IsInRing())]
    xyz_arr = np.array(xyz_arr)

    # get the atom types of atoms
    atom_type = [atom.GetSymbol() for atom in mol_from_smiles.GetAtoms()]
    return True, xyz_arr, atom_type


def precursor_calculator(precursor_type, mass):
    if precursor_type == "[M+H]+":
        return mass + 1.007276
    elif precursor_type == "[M+Na]+":
        return mass + 22.989218
    elif precursor_type == "[2M+H]+":
        return 2 * mass + 1.007276
    elif precursor_type == "[M-H]-":
        return mass - 1.007276
    elif precursor_type == "[M+H-H2O]+":
        return mass - 17.0038370665
    elif precursor_type == "[M+2H]2+":
        return mass / 2 + 1.007276
    else:
        raise ValueError("Unsupported precursor type: {}".format(precursor_type))


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = []
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.config = load_config(os.path.join(file_path, "3dmolms_config.json"))

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        print("HERE IN PREPROCESS INIT")
        output_SMILES_out_config = pb_utils.get_output_config_by_name(
            model_config, "SMILES_out"
        )
        self.output_SMILES_out_dtype = pb_utils.triton_string_to_numpy(
            output_SMILES_out_config["data_type"]
        )
        output_precursor_type_out_config = pb_utils.get_output_config_by_name(
            model_config, "precursor_type_out"
        )
        self.output_precursor_type_out_dtype = pb_utils.triton_string_to_numpy(
            output_precursor_type_out_config["data_type"]
        )
        idx_base_out_config = pb_utils.get_output_config_by_name(
            model_config, "idx_base_out"
        )
        self.output_idx_base_out_dtype = pb_utils.triton_string_to_numpy(
            idx_base_out_config["data_type"]
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            # Extract input tensors by their names
            SMILES_in = pb_utils.get_input_tensor_by_name(request, "SMILES_in")
            precursor_type_in = pb_utils.get_input_tensor_by_name(
                request, "precursor_type_in"
            )
            collision_energy_in = pb_utils.get_input_tensor_by_name(
                request, "collision_energy_in"
            )

            SMILES_in = SMILES_in.as_numpy().tolist()
            precursor_type_in = precursor_type_in.as_numpy().tolist()
            collision_energy_in = collision_energy_in.as_numpy().tolist()

            print("SMILES_in", SMILES_in)
            print("precursor_type_in", precursor_type_in)
            print("collision_energy_in", collision_energy_in)

            SMILES_in = [x[0].decode("utf-8") for x in SMILES_in]
            SMILES_in = SMILES_in[0]

            precursor_type_in = [x[0].decode("utf-8") for x in precursor_type_in]
            precursor_type_in = precursor_type_in[0]

            collision_energy_in = collision_energy_in[0][0]

            x_data, env_data, idx_base_data = pre_process(
                SMILES_in, precursor_type_in, collision_energy_in, self.config
            )

            # Create output tensors
            SMILES_out = pb_utils.Tensor(
                "SMILES_out", x_data.astype(self.output_SMILES_out_dtype)
            )
            precursor_type_out = pb_utils.Tensor(
                "precursor_type_out",
                env_data.astype(self.output_precursor_type_out_dtype),
            )
            idx_base_out = pb_utils.Tensor(
                "idx_base_out", idx_base_data.astype(self.output_idx_base_out_dtype)
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[SMILES_out, precursor_type_out, idx_base_out]
                )
            )
            # raise ValueError("size of x_data: {} size of env_data: {} size of idx_base_data: {}".format(x_data.shape, env_data.shape, idx_base_data.shape))

        return responses

    def finalize(self):
        """Clean up resources."""
        pass
