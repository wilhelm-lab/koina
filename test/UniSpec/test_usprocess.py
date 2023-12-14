from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
import requests
from pathlib import Path
import re

# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")

mdicum = {
    1: 'Acetyl',
    4: 'Carbamidomethyl',
    28: 'Gln->pyro-Glu',
    27: 'Glu->pyro-Glu',
    35: 'Oxidation',
    21: 'Phospho',
    26: 'Pyro-carbamidomethyl',
    #4: 'CAM'
}
rev_mdicum = {n:m for m,n in mdicum.items()}

def str2dat(label):
    seq,other = label.split('/')
    [charge, mods, ev, nce] = other.split('_')
    return (seq,mods,int(charge),float(ev[:-2]),float(nce[3:]))

def label2modseq(labels):
    modseqs = []
    charges = []
    nces = []
    for label in labels:
        
        seq, mod, charge, ev, nce = str2dat(label)
        charges.append(int(charge))
        nces.append(float(nce))

        mseq = seq
        Mstart = mod.find('(') if mod != '0' else 1
        modamt = int(mod[0:Mstart])
        if modamt > 0:
            hold = [re.sub('[()]', '', n) for n in mod[Mstart:].split(")(")]
            hold.reverse()
            for n in hold:
                [pos, aa, modtyp] = n.split(',')
                pos = int(pos)
                assert seq[pos] == aa
                assert "Carbamidomethyl" in rev_mdicum.keys(), print(rev_mdicum.keys())
                mseq = list(mseq)[:pos+1] + list("[UNIMOD:%d]"%rev_mdicum[modtyp]) + list(mseq)[pos+1:]
                mseq = "".join(mseq)
        modseqs.append(mseq)
    Modseqs = np.array(modseqs)[:,None].astype(np.object_)
    Charges = np.array(charges)[:,None].astype(np.int32)
    Nces = np.array(nces)[:,None].astype(np.float32)

    return Modseqs, Charges, Nces

def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 200


def test_available_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)


def test_inference():
    
    labels = open("test/UniSpec/labels_input.txt").read().split("\n")
    SEQUENCES, charge, ces = label2modseq(labels)
    instr = np.array(50*['QE'])[:,None].astype(np.object_)

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    
    in_pep_seq = grpcclient.InferInput("peptide_sequences", SEQUENCES.shape, "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    in_charge = grpcclient.InferInput("precursor_charges", charge.shape, "INT32")
    in_charge.set_data_from_numpy(charge)

    in_ces = grpcclient.InferInput("collision_energies", ces.shape, "FP32")
    in_ces.set_data_from_numpy(ces)

    in_instr = grpcclient.InferInput("instrument_types", instr.shape, "BYTES")
    in_instr.set_data_from_numpy(instr)
    
    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge, in_ces, in_instr],
        outputs=[
            grpcclient.InferRequestedOutput("intensities"),
            grpcclient.InferRequestedOutput("mz"),
            grpcclient.InferRequestedOutput("annotation")
        ],
    )

    intensities = result.as_numpy("intensities")
    mz = result.as_numpy("mz")
    ann = result.as_numpy("annotation")

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/UniSpec/test_output_tensor.npy"),
        rtol=0,
        atol=1e-3,
    )
    assert np.allclose(
        mz,
        np.load("test/UniSpec/test_output_mz.npy"),
        rtol=0,
        atol=1e-3,
    )
