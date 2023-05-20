import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    server_url = "serving:8500"
    model_name = "ms2pip_2021_HCD"
    out_layer = "intensities"
    peptides = [["ACDEK"], ["AAAAAAAAAAAAA"]]
    batch_size = len(peptides)
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput("peptide_sequences", [batch_size, 1], "BYTES"))
    inputs.append(grpcclient.InferInput("precursor_charge", [batch_size, 1], "INT16"))
    peptide_seq_in = np.array([i for i in peptides], dtype=np.object_)
    precursor_charge_in = np.array([[2],[3]], dtype=np.int16)
    print("len: " + str(len(inputs)))
    inputs[0].set_data_from_numpy(peptide_seq_in)
    inputs[1].set_data_from_numpy(precursor_charge_in)

    outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    end = time.time()
    print("Time: " + str(end - start))

    print("Result")
    print(result.as_numpy(out_layer))
    tt = result.as_numpy(out_layer)
    print(tt.reshape((-1, 29)))
