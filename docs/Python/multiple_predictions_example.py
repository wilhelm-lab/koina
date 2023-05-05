import numpy as np
import time
import tritonclient.grpc as grpcclient

server_url = "serving:8500"
model_name_ms = "Prosit_2019_intensity"
out_layer_ms = "intensities"
inputs_ms = []
outputs_ms = []

model_name_rt = "Deeplc_Triton_ensemble"
out_layer_rt = "dense_323"
inputs_rt = []
outputs_rt = []

batch_size = 2


triton_client = grpcclient.InferenceServerClient(url=server_url)

inputs_ms.append(grpcclient.InferInput("peptide_sequences", [batch_size, 1], "BYTES"))
inputs_ms.append(
    grpcclient.InferInput("collision_energies", [batch_size, 1], "FP32")
)
inputs_ms.append(
    grpcclient.InferInput("precursor_charge", [batch_size, 1], "INT32")
)

inputs_rt.append(grpcclient.InferInput("peptide_sequences", [batch_size, 1], "BYTES"))


# Create the data for the two input tensors. Initialize the first
# to unique integers and the second to all ones.
peptide_seq_in = np.array(
    [[b"AAAAAKAC[UNIMOD:4]"] for i in range(0, batch_size)], dtype=np.object_
)
ce_in = np.array([[0.25] for i in range(0, batch_size)], dtype=np.float32)
precursor_charge_in = np.array([[1] for i in range(0, batch_size)], dtype=np.int32)

# Initialize the data
inputs_ms[0].set_data_from_numpy(peptide_seq_in)
inputs_ms[1].set_data_from_numpy(ce_in)
inputs_ms[2].set_data_from_numpy(precursor_charge_in)

inputs_rt[0].set_data_from_numpy(peptide_seq_in)

outputs_ms.append(grpcclient.InferRequestedOutput(out_layer_ms))
outputs_rt.append(grpcclient.InferRequestedOutput(out_layer_rt))


start = time.time()
result_ms = triton_client.infer(model_name_ms, inputs=inputs_ms, outputs=outputs_ms)
result_rt = triton_client.infer(model_name_rt, inputs=inputs_rt, outputs=outputs_rt)

end = time.time()
print("Time: " + str(end - start))

print("Result")
print(result_ms.as_numpy(out_layer_ms))
print(result_rt.as_numpy(out_layer_rt))
