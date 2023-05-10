import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    server_url = "serving:8500"
    model_name = "Deeplc_Triton_ensemble"
    out_layer = "dense_323"
    batch_size = 1
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput("peptide_sequences", [4, 1], "BYTES"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input_141 = np.array(
        [["ACDEK"], ["ACDEFGK"], ["ACDEFGHIKLR"], ["ACDEFGHIKLMNPK"]],
        dtype=np.object_,
    )

    # Initialize the data
    inputs[0].set_data_from_numpy(input_141)

    outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    end = time.time()
    print("Time: " + str(end - start))

    print("Result")
    print(result.as_numpy(out_layer))
