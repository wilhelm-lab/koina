import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    # server_url = 'localhost:8500'
    server_url = "eubic2023.external.msaid.io:8500"
    model_name = "ms2pip_ensemble"
    out_layer = "model_20210416_HCD2021_B_output"
    peptides = [["ACDEK/2"], ["AAAAAAAAAAAAA/3"]]
    batch_size = len(peptides)
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput("proforma_ensemble", [batch_size, 1], "BYTES"))
    peptide_seq_in = np.array([i for i in peptides], dtype=np.object_)
    print("len: " + str(len(inputs)))
    inputs[0].set_data_from_numpy(peptide_seq_in)

    outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    end = time.time()
    print("Time: " + str(end - start))

    print("Result")
    print(result.as_numpy(out_layer))
    tt = result.as_numpy(out_layer)
    print(tt.reshape((-1, 29)))
