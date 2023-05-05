import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    server_url = "serving:8500"
    model_name = "Prosit_2019_intensity"
    out_layer1 = "intensities"
    out_layer2 = "mz"
    batch_size = 5
    inputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput("peptide_sequences", [batch_size, 1], "BYTES"))
    inputs.append(
        grpcclient.InferInput("collision_energies", [batch_size, 1], "FP32")
    )
    inputs.append(
        grpcclient.InferInput("precursor_charge", [batch_size, 1], "INT32")
    )

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    peptide_seq_in = np.array(
        [["AAAAAKAKM[UNIMOD:35]"] for i in range(0, batch_size)], dtype=np.object_
    )
    ce_in = np.array([[25] for i in range(0, batch_size)], dtype=np.float32)
    precursor_charge_in = np.array([[2] for i in range(0, batch_size)], dtype=np.int32)

    # Initialize the data
    print("len: " + str(len(inputs)))
    inputs[0].set_data_from_numpy(peptide_seq_in)
    inputs[1].set_data_from_numpy(ce_in)
    inputs[2].set_data_from_numpy(precursor_charge_in)

    outputs = [
        grpcclient.InferRequestedOutput(out_layer1),
        grpcclient.InferRequestedOutput(out_layer2),
    ]

    start = time.time()
    # result = triton_client.infer(model_name,inputs=[inputs[0]], outputs=outputs)
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    end = time.time()
    print("Time: " + str(end - start))

    print("Result")
    print(np.round(result.as_numpy(out_layer1), 1))
    print(np.round(result.as_numpy(out_layer2), 1))
