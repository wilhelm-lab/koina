import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    server_url = "localhost:8500"
    model_name = "Deeplc_rt"
    out_layer = "dense_323"
    batch_size = 1
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput("input_141", [batch_size, 60, 6], "FP32"))
    inputs.append(grpcclient.InferInput("input_142", [batch_size, 30, 6], "FP32"))
    inputs.append(grpcclient.InferInput("input_143", [batch_size, 55], "FP32"))
    inputs.append(grpcclient.InferInput("input_144", [batch_size, 60, 20], "FP32"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input_141 = np.ones([1, 60, 6], dtype=np.float32)
    input_142 = np.ones([1, 30, 6], dtype=np.float32)
    input_143 = np.ones([1, 55], dtype=np.float32)
    input_144 = np.ones([1, 60, 20], dtype=np.float32)

    # Initialize the data
    print("len: " + str(len(inputs)))
    inputs[0].set_data_from_numpy(input_141)
    inputs[1].set_data_from_numpy(input_142)
    inputs[2].set_data_from_numpy(input_143)
    inputs[3].set_data_from_numpy(input_144)

    outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    end = time.time()
    print("Time: " + str(end - start))

    print("Result")
    print(result.as_numpy(out_layer))
