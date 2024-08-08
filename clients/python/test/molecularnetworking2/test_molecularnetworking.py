#from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from pathlib import Path
import requests
import numpy as np

# # To ensure MODEL_NAME == test_<filename>.py
# MODEL_NAME = Path(__file__).stem.replace("test_", "")


# def test_available_http():
#     req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
#     assert req.status_code == 200


def preprocess_example(mz_array1, intensity_array1, mz_array2, intensity_array2, precursor_mass_diff, precursor_mz1, precursor_mz2, max_seq_len=100):
    """
    Preprocesses a single example by padding sequences and converting them to tensors.

    Args:
        mz_array1 (list): List of mass-to-charge ratio values for array 1.
        intensity_array1 (list): List of intensity values for array 1.
        mz_array2 (list): List of mass-to-charge ratio values for array 2.
        intensity_array2 (list): List of intensity values for array 2.
        precursor_mass_diff (float): Precursor mass difference.
        precursor_mz1 (float): Precursor mass-to-charge ratio 1.
        precursor_mz2 (float): Precursor mass-to-charge ratio 2.
        max_seq_len (int): Maximum sequence length for padding.

    Returns:
        tuple: Preprocessed mz_array1, intensity_array1, neutral_loss_1, mz_array2, intensity_array2, neutral_loss_2, and precursor_info as tensors.
    """
    try:
        def pad_sequence(sequence, max_length=max_seq_len):
            if len(sequence) < max_length:
                padding_length = max_length - len(sequence)
                sequence = np.pad(sequence, (0, padding_length), 'constant')
            return sequence[:max_length]

        mz_array1 = pad_sequence(mz_array1)
        intensity_array1 = pad_sequence(intensity_array1)
        neutral_loss_1 = precursor_mz1 - np.array(mz_array1)

        mz_array2 = pad_sequence(mz_array2)
        intensity_array2 = pad_sequence(intensity_array2)
        neutral_loss_2 = precursor_mz2 - np.array(mz_array2)

        precursor_info = np.array([precursor_mass_diff, precursor_mz1, precursor_mz2])

        return mz_array1, intensity_array1, neutral_loss_1, mz_array2, intensity_array2, neutral_loss_2, precursor_info
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

def test_available_grpc():
    SERVER_GRPC = "localhost:8500"
    MODEL_NAME = "edit_distance"

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)

def test_inference():
    bareserver = "localhost:8501"
    SERVER_HTTP = "http://localhost:8501"
    MODEL_NAME = "edit_distance"

    url = f"{SERVER_HTTP}/v2/models/{MODEL_NAME}/infer"

    triton_client = httpclient.InferenceServerClient(url=bareserver)

    # Check if the server is live
    if not triton_client.is_server_live():
        print("Triton server is not live!")
        exit(1)

    mz_array1 = [100, 200, 300] 
    intensity_array1 = [10, 20, 30]  
    mz_array2 = [150, 250, 350]  
    intensity_array2 = [15, 25, 35]  
    precursor_mass_diff = 20.0  
    precursor_mz1 = 500.0  
    precursor_mz2 = 600.0  

    mz_array1, intensity_array1, neutral_loss_1, mz_array2, intensity_array2, neutral_loss_2, precursor_info = preprocess_example(
        mz_array1, intensity_array1, mz_array2, intensity_array2, precursor_mass_diff, precursor_mz1, precursor_mz2
    )

    input_data = {
        "mz_array1": mz_array1,
        "intensity_array1": intensity_array1,
        "neutral_loss_1": neutral_loss_1,
        "mz_array2": mz_array2,
        "intensity_array2": intensity_array2,
        "neutral_loss_2": neutral_loss_2,
        "precursor_information": precursor_info
    }
    
    output = httpclient.InferRequestedOutput("prediction")

    response = triton_client.infer(MODEL_NAME, inputs=[input_data], outputs=[output])

    output_data = response.as_numpy("prediction")

    print(output_data)


def main():
    test_inference()
    #test_available_grpc()

if __name__ == "__main__":
    main()