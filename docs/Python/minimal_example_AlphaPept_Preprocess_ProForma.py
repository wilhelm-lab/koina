import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    server_url = "213.239.214.190:8504"
    model_name = "AlphaPept_Preprocess_ProForma"
    out_layer = "encoded_mod_feature:0"
    batch_size = 100
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput("peptide_sequences", [batch_size, 1], "BYTES"))

    peptide_seq_in = np.array(
        [["AAAAAKAK"] for i in range(0, batch_size)], dtype=np.object_
    )
    ce_in = np.array([[0.25] for i in range(0, batch_size)], dtype=np.float32)
    precursor_charge_in = np.array([[1] for i in range(0, batch_size)], dtype=np.int32)

    # Initialize the data
    inputs[0].set_data_from_numpy(peptide_seq_in)

    outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    end = time.time()
    print("Time: " + str(end - start))

    print("Result")
    print(result.as_numpy(out_layer))
