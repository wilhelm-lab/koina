import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    server_url = "serving:8500"
    model_name = "fragment_mz"
    batch_size = 100
    out_layer = "fragment_mz"
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput("ProForma", [batch_size, 1], "BYTES"))
    peptide_seq_in = np.array(
        [["A" * 120] for _ in range(0, batch_size)], dtype=np.object_
    )
    inputs[-1].set_data_from_numpy(peptide_seq_in)

    inputs.append(grpcclient.InferInput("charges", [4], "INT32"))
    charge_in = np.array([1, 2, 3, 4], dtype=np.int32)
    inputs[-1].set_data_from_numpy(charge_in)

    inputs.append(grpcclient.InferInput("ion_series", [3], "BYTES"))
    ion_series_in = np.array(["b", "y", "a"], dtype=np.object_)
    inputs[-1].set_data_from_numpy(ion_series_in)

    outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    end = time.time()
    print("Time: " + str(end - start))

    print("Result")
    print(result.as_numpy(out_layer).shape)
