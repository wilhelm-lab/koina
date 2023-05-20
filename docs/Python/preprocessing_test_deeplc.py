import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    server_url = "serving:8500"
    model_name = "Deeplc_Triton_ensemble"
    out_layers = ["single_ac", "peptides_in:0", "diamino_ac", "general_features"]
    out_layers = ["dense_323"]
    batch_size = 4
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput("peptide_sequences", [4, 1], "BYTES"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    peptide_seq_in = np.array(
        [[b"LGGNEQVTR"], [b"GAGSSEPVTGLDAK"], [b"VEATFGVDESNAK"], [b"LFLQFGAQGSPFLK"]],
        dtype=np.object_,
    )
    # peptide_seq_in = np.array([ [b"LFLQFGAQGSPFLK"]], dtype=np.object_)

    # Initialize the data
    print("len: " + str(len(inputs)))
    inputs[0].set_data_from_numpy(peptide_seq_in)

    for out_layer in out_layers:
        outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    end = time.time()
    print("Time: " + str(end - start))

    print("Result")
    for out_layer in out_layers:
        print(result.as_numpy(out_layer))
        print(result.as_numpy(out_layer).shape)
