import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == '__main__':
    server_url = '213.239.214.190:8504'
    model_name = "AlphaPept_Preprocess_mod"
    out_layer = 'output:0'
    batch_size = 100
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput('mods:0', [batch_size,1], "BYTES"))
    inputs.append(grpcclient.InferInput('mod_sites:0', [batch_size,1], "BYTES"))
    inputs.append(grpcclient.InferInput('nAA:0',[batch_size,1],"INT32"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    mods = np.array([ ["Oxidation@M;Oxidation@M"] for i in range (0,batch_size) ], dtype=np.object_)
    mod_sites = np.array([ ["2;2"] for i in range (0,batch_size) ], dtype=np.object_)
    nAA = np.array([ [23] for i in range (0,batch_size) ], dtype=np.int32)

    # Initialize the data
    print("len: "  + str(len(inputs)))
    inputs[0].set_data_from_numpy(mods)
    inputs[1].set_data_from_numpy(mod_sites)
    inputs[2].set_data_from_numpy(nAA)
    
    outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    result = triton_client.infer(model_name,inputs=inputs, outputs=outputs)
    end = time.time()
    print( "Time: " + str(end - start))   

    print('Result')
    print(result.as_numpy(out_layer))
