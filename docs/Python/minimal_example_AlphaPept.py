import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    server_url = "serving:8500"
    model_name = "AlphaPept_ms2_generic"
    out_layer = "intensities"
    # model_name = "AlphaPept_Preprocess_ProForma"
    # out_layer = 'encoded_seq:0'
    batch_size = 100
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput("peptide_sequences", [batch_size, 1], "BYTES"))
    inputs.append(grpcclient.InferInput("collision_energies", [batch_size, 1], "INT32"))
    inputs.append(grpcclient.InferInput("precursor_charge", [batch_size, 1], "INT32"))
    inputs.append(grpcclient.InferInput("instrument_types", [batch_size, 1], "INT64"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    peptide_seq_in = np.array(
        [["AAAAAKAKM[UNIMOD:21]"] for i in range(0, batch_size)], dtype=np.object_
    )
    ce_in = np.array([[25] for i in range(0, batch_size)], dtype=np.int32)
    precursor_charge_in = np.array([[2] for i in range(0, batch_size)], dtype=np.int32)
    instrument_in = np.array([[1] for i in range(0, batch_size)], dtype=np.int64)

    # Initialize the data
    print("len: " + str(len(inputs)))
    inputs[0].set_data_from_numpy(peptide_seq_in)
    inputs[1].set_data_from_numpy(ce_in)
    inputs[2].set_data_from_numpy(precursor_charge_in)
    inputs[3].set_data_from_numpy(instrument_in)

    outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    # result = triton_client.infer(model_name,inputs=[inputs[0]], outputs=outputs)
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    end = time.time()
    print("Time: " + str(end - start))

    print("Result")
    print(result.as_numpy(out_layer))
