import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == '__main__':
    server_url = '213.239.214.190:8500'
    model_name = "Prosit_2019_intensity_triton"
    out_layer = 'out/Reshape:1'
    batch_size = 100
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput('peptides_in_str:0', [batch_size,1], "BYTES"))
    inputs.append(grpcclient.InferInput('collision_energy_in:0',[batch_size,1],"FP32"))
    inputs.append(grpcclient.InferInput('precursor_charge_in_int:0',[batch_size,1],"INT32"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    peptide_seq_in = np.array([ ["AAAAAKAK"] for i in range (0,batch_size) ], dtype=np.object_)
    ce_in = np.array([ [0.25] for i in range(0,batch_size) ], dtype=np.float32)
    precursor_charge_in = np.array([ [1] for i in range (0,batch_size) ], dtype=np.int32)

    # Initialize the data
    print("len: "  + str(len(inputs)))
    inputs[0].set_data_from_numpy(peptide_seq_in)
    inputs[1].set_data_from_numpy(ce_in)
    inputs[2].set_data_from_numpy(precursor_charge_in)
    
    outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    result = triton_client.infer(model_name,inputs=inputs, outputs=outputs)
    end = time.time()
    print( "Time: " + str(end - start))   

    print('Result')
    print(result.as_numpy(out_layer))
