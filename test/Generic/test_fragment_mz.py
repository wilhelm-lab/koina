from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
import requests
from pathlib import Path

# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")

def ppm_error(arr1, arr2):
    return (np.abs(arr1 - arr2) / arr1) * 1_000_000


def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 200


def test_available_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)


def test_inference():
    """
    Verify fragment mass calculation with https://proteomicsresource.washington.edu/cgi-bin/fragment.cgi
    """
    out_layer = "output_fragmentmz"
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("ProForma", [5], "BYTES")
    in_pep_seq.set_data_from_numpy(
        np.array(["PEPTIDE" for _ in range(0, 5)], dtype=np.object_)
    )

    in_charge = grpcclient.InferInput("charges", [7], "INT32")
    in_charge.set_data_from_numpy(np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32))

    in_ion_series = grpcclient.InferInput("ion_series", [6], "BYTES")
    in_ion_series.set_data_from_numpy(
        np.array(["a", "b", "c", "x", "y", "z"], dtype=np.object_)
    )

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge, in_ion_series],
        outputs=[grpcclient.InferRequestedOutput(out_layer)],
    ).as_numpy(out_layer)

    assert result.shape == (5, 6, 7, 32)

    # Fragment ion masses verified at https://proteomicsresource.washington.edu/cgi-bin/fragment.cgi
    # a ions charge 1
    assert np.all(
        ppm_error(
            result[0, 0, 1, :6],
            np.array(
                [
                    70.065126,
                    199.107719,
                    296.160483,
                    397.208161,
                    510.292225,
                    625.319168,
                ]
            ),
        )
        < 0.01
    )

    # b ions charge 1
    assert np.all(
        ppm_error(
            result[0, 1, 1, :6],
            np.array(
                [
                    98.060040,
                    227.102633,
                    324.155397,
                    425.203076,
                    538.287140,
                    653.314083,
                ]
            ),
        )
        < 0.01
    )

    # c ions charge 1
    assert np.all(
        ppm_error(
            result[0, 2, 1, :6],
            np.array(
                [
                    115.086589,
                    244.129183,
                    341.181946,
                    442.229625,
                    555.313689,
                    670.340632,
                ]
            ),
        )
        < 0.01
    )

    # x ions charge 1
    assert np.all(
        ppm_error(
            result[0, 3, 1, :6],
            np.array(
                [
                    174.039699,
                    289.066642,
                    402.150706,
                    503.198384,
                    600.251148,
                    729.293741,
                ]
            ),
        )
        < 0.01
    )

    # y ions charge 1
    assert np.all(
        ppm_error(
            result[0, 4, 1, :6],
            np.array(
                [
                    148.060434,
                    263.087377,
                    376.171441,
                    477.219120,
                    574.271884,
                    703.314477,
                ]
            ),
        )
        < 0.01
    )

    # z ions charge 1
    assert np.all(
        ppm_error(
            result[0, 5, 1, :6],
            np.array(
                [
                    131.033885,
                    246.060828,
                    359.144892,
                    460.192571,
                    557.245335,
                    686.287928,
                ]
            ),
        )
        < 0.01
    )

    # a ions charge 2
    assert np.all(
        ppm_error(
            result[0, 0, 2, :6],
            np.array(
                [
                    35.536201,
                    100.057498,
                    148.583880,
                    199.107719,
                    255.649751,
                    313.163222,
                ]
            ),
        )
        < 0.01
    )

    # b ions charge 3
    assert np.all(
        ppm_error(
            result[0, 1, 3, :6],
            np.array(
                [
                    33.358198,
                    76.372395,
                    108.723317,
                    142.405876,
                    180.100564,
                    218.442879,
                ]
            ),
        )
        < 0.01
    )

    # c ions charge 3
    assert np.all(
        ppm_error(
            result[0, 2, 3, :6],
            np.array(
                [
                    39.033714,
                    82.047912,
                    114.398833,
                    148.081393,
                    185.776081,
                    224.118395,
                ]
            ),
        )
        < 0.01
    )

    # x ions charge 4
    assert np.all(
        ppm_error(
            result[0, 3, 4, :6],
            np.array(
                [
                    44.265382,
                    73.022118,
                    101.293134,
                    126.555053,
                    150.818244,
                    183.078893,
                ]
            ),
        )
        < 0.01
    )

    # y ions charge 5
    assert np.all(
        ppm_error(
            result[0, 4, 5, :6],
            np.array(
                [
                    30.417908,
                    53.423297,
                    76.040109,
                    96.249645,
                    115.660198,
                    141.468717,
                ]
            ),
        )
        < 0.01
    )

    # z ions charge 6
    assert np.all(
        ppm_error(
            result[0, 5, 6, :6],
            np.array(
                [
                    22.678378,
                    41.849535,
                    60.696879,
                    77.538159,
                    93.713619,
                    115.220718,
                ]
            ),
        )
        < 0.01
    )
