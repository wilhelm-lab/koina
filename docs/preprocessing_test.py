import numpy as np
import time
import tritonclient.grpc as grpcclient

if __name__ == '__main__':
    server_url = 'localhost:8502'
    model_name = "Deeplc_Preprocess_unmod"
    out_layer = 'stripped_peptide'
    batch_size = 1
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(url=server_url)

    inputs.append(grpcclient.InferInput('peptides_in_str:0', [batch_size,1], "BYTES"))
    
    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    peptide_seq_in = np.array([ ["EM[UNIMOD:35]EVEES[UNIMOD:21]PEK"] for i in range (0,batch_size) ], dtype=np.object_)    


    
    # Initialize the data
    print("len: "  + str(len(inputs)))
    inputs[0].set_data_from_numpy(peptide_seq_in)
    
    
    outputs.append(grpcclient.InferRequestedOutput(out_layer))

    start = time.time()
    result = triton_client.infer(model_name,inputs=inputs, outputs=outputs)
    end = time.time()
    print( "Time: " + str(end - start))   

    print('Result')
    print(result.as_numpy(out_layer))
