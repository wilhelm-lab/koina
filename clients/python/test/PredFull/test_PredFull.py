from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
from pathlib import Path
import requests
import time

# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")


def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 200


def test_available_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)


def test_inference():
    SEQUENCES = np.array(
        [
            ["AA"],
            ["PEPTIPEPTIPEPTIPEPTIPEPTIPEPT"],
            ["RHKDESTNQCGPAVILMFYW"],
            ["RHKDESTNQCGPAVILM[UNIMOD:35]FYW"],
        ],
        dtype=np.object_,
    )
    SEQUENCES_copy = SEQUENCES
    for i in range(249):
        for seq in SEQUENCES_copy:
            SEQUENCES = np.append(SEQUENCES, [seq], axis=0)
    len_s = len(SEQUENCES)

    charge = np.array([[i % 6 + 1] for i in range(len_s)], dtype=np.int32)
    nce = np.array([[(19 + i) % 40 + 1] for i in range(len_s)], dtype=np.float32)
    fragmentation_type = np.array(
        [["HCD" if i % 2 != 0 else "CID"] for i in range(len_s)], dtype=np.object_
    )

    start = time.time()
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptide_sequences", [len_s, 1], "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    in_charge = grpcclient.InferInput("precursor_charges", [len_s, 1], "INT32")
    in_charge.set_data_from_numpy(charge)

    in_nce = grpcclient.InferInput("collision_energies", [len_s, 1], "FP32")
    in_nce.set_data_from_numpy(nce)

    in_frag = grpcclient.InferInput("fragmentation_types", [len_s, 1], "BYTES")
    in_frag.set_data_from_numpy(fragmentation_type)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge, in_nce, in_frag],
        outputs=[
            grpcclient.InferRequestedOutput("mzs"),
            grpcclient.InferRequestedOutput("intensities"),
        ],
    )

    fragmentmz = result.as_numpy("mzs")
    intensities = result.as_numpy("intensities")
    end = time.time()
    print(fragmentmz[0:5, :])
    print(intensities[0:5, :])
    print(end - start)

    for i, (test_mzs, test_ints) in enumerate(
        zip(
            [
                "AA1mzsCID20.npy",
                "PEPTIPEPTIPEPTIPEPTIPEPTIPEPT2mzsHCD21.npy",
                "RHKDESTNQCGPAVILMFYW3mzsCID22.npy",
                "RHKDESTNQCGPAVILM(ox)FYW4mzsHCD23.npy",
            ],
            [
                "AA1intsCID20.npy",
                "PEPTIPEPTIPEPTIPEPTIPEPTIPEPT2intsHCD21.npy",
                "RHKDESTNQCGPAVILMFYW3intsCID22.npy",
                "RHKDESTNQCGPAVILM(ox)FYW4intsHCD23.npy",
            ],
        )
    ):
        # get predicted arrays
        pred_mzs = fragmentmz[i, :]
        pred_ints = intensities[i, :]
        pred_mzs = pred_mzs[pred_mzs != -1]
        pred_ints = pred_ints[pred_ints != -1]
        pred_ints /= 1000

        # get elements in test loaded array
        ground_truth_mzs = np.load("test/PredFull/numpy_arrays/" + test_mzs)
        p_indices = np.where(np.isin(pred_mzs, ground_truth_mzs))[0]
        gt_indices = np.where(np.isin(ground_truth_mzs, pred_mzs))[0]

        print(
            np.max(
                abs(
                    pred_ints[p_indices]
                    - np.load("test/PredFull/numpy_arrays/" + test_ints)[gt_indices]
                )
            )
        )
        assert np.allclose(
            pred_ints[p_indices],
            np.load("test/PredFull/numpy_arrays/" + test_ints)[gt_indices],
            rtol=0,
            atol=1e-1,
            equal_nan=True,
        )
