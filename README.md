# Prosit serving

# Common difficulties/errors

## Doing the batching wrong
The triton version is picky about the version of CUDA and Nvidia drivers. Therefore you need to cross-reference the version numbers.

When you implement the triton business logic, you also need to care of the batching as well.
For instance this is how you receive an input layer from the user in triton business logic.

```python
peptide_in = pb_utils.get_input_tensor_by_name(request, "peptides_in_str:0")
peaks_in = pb_utils.get_input_tensor_by_name(request, "peaks_in:0").as_numpy().tolist()
```

If the batching is enabled. (Enabled means the max_batch parameter is set greater than 0). What you would receive from the user is two dimensional array if your input was one dimensional. Therefore you need to iterate the list like a two dimensional array below:

```python
for batch in peptide_in:
  for peptide in batch:
    peptide_lengths.append(len(peptide))

```
If the batch parsing is implemented wrong in the ensemble model, it will complain about the batch sizes are wrong the typical error message is:

```
tritonclient.utils.InferenceServerException: [StatusCode.INVALID_ARGUMENT] in ensemble 'Prosit_2019_intensity_triton', input 'peptides_in:0' batch size does not match other inputs for 'Prosit_2019_intensity'  
```

# Starting Triton server

there is a script in the repo called `run.sh` you can invoke the script by calling it with a parameter `--tr`: namely: ```./run.sh --tr```. The current triton server docker image is "nvcr.io/nvidia/tritonserver:22.03-py3"

# Starting Triton client

The triton client grpc SDK comes with also a docker image. The name of the docker image is: 
`nvcr.io/nvidia/tritonserver:22.03-py3-sdk`. One way to use this image is to run docker run/exec command to execute bash and invoke the script inside of the image. Or you can use this image to create another image on top, which can override the "entrypoint". The easiest way is to start the image with:
```docker run -it nvcr.io/nvidia/tritonserver:22.03-py3-sdk bash``` Once you execute this command you will be inside of the container. In that you have all the SDK and libraries you need. The simple grpc client that is used to test the triton-server is committed in this repo with the name  `simple_grpc_infer_client.py`. You can copy the script into running container by:
`docker cp simple_grpc_infer_client.py <ID_OF_CONTAINER>:<PATH_IN_CONTAINER>`

Then inside the container you can simply execute the script.

# Triton ensemble pipeline for Prosit 2019 intensity model

The triton inference server allows one to implement custom business logic and bind it to the actual machine learning model that you saved as a protobuf. We used this feature on the preprossing and postprocessing of the prosit 2019 model. The way we implemented it is creating two custom preprocssing business logic units and one post processing unit.

The structure of the business logic looks like one below. The numbered directory tells the version of the model.
```
.
├── 1
│   ├── model.py
└── config.pbtxt
```
The two preprocessing units have names: Prosit_Preprocess_charge, and Prosit_Preprocess_peptide. The postprocessing unit has a name of Prosit_2019_intensity_postprocess

## Preprocessing: Prosit_Preprocess_charge
    This unit converts array of charges into one-hot encoded matrix of charges
## Preprocessing: Prosit_Preprocess_peptide
    This unit converts string of peptide sequences into fixed length of 30 sequence of numbers
## Postprocessing: Prosit_2019_intensity_postprocess
    This unit sets -1 to the elements of peaks that are not valid

# Config.pbtxt structure

For config.pbtxt you need to describe the Input and Output tensors. For example in postprocessing unit below. Describes what input it takes: it takes peptide_in string of peptide sequences, precursor charge that is output of Prosit_Preprocess_charge business logic unit and peaks_in which is the output tensor of Prosit_2019_intensity model. Then as for the output it gives you back the postprocessed tensors with the same length as the Prosit_2019_intensity model.

```
name: "Prosit_2019_intensity_postprocess"
max_batch_size: 7000
backend: "python"

input[
  {
   name: 'peptides_in:0',
   data_type: TYPE_STRING,
   dims: [-1]
  },
  {
    name: 'precursor_charge_in:0',
    data_type: TYPE_FP32,
    dims: [6],
	},
  {
   name: 'peaks_in:0',
   data_type: TYPE_FP32,
   dims: [174]
  }
]
output [
 {
   name: 'out/Reshape:1',
   data_type: TYPE_FP32,
   dims: [174]
 }
]

```

# Ensemble scheduling model structure
The ensemble model would take the input/output tensors like any other models. The only difference is in the ensemble_scheduling property you can now link different models together. The code snippet below is the final triton model. Which is called Prosit_2019_intensity_triton. This model passes the input tensors that are sent by the user to the preprocessing models and takes the output and sends it to the actual Prosit_2019_intensity ML model. Afterwards it gets the output from the Prosit_2019_intensity model and sends it to the post processing unit and delivers the output to the user with the name "out/Reshape:1"

```
name: "Prosit_2019_intensity_triton"
max_batch_size: 7000
platform: "ensemble"
input [
  {
   name: 'peptides_in_str:0',
   data_type: TYPE_STRING,
   dims: [-1]
  },
  {
    name: 'precursor_charge_in_int:0',
    data_type: TYPE_INT32,
    dims: [1],
  },
  {
    name: 'collision_energy_in:0',
    data_type: TYPE_FP32,
    dims: [1],
  }
]
output [
  {
   name: 'out/Reshape:1',
   data_type: TYPE_FP32,
   dims: [174]
  }
]
ensemble_scheduling {
  step [
     {
      model_name: "Prosit_Preprocess_charge"
      model_version: 1
      input_map {
        key: "precursor_charge_in_int:0"
        value: "precursor_charge_in_int:0"
      },
      output_map {
        key: "precursor_charge_in:0"
        value: "precursor_charge_in_preprocessed:0"
      }
    },
    {
      model_name: "Prosit_Preprocess_peptide"
      model_version: 1
      input_map {
        key: "peptides_in_str:0"
        value: "peptides_in_str:0"
      },
      output_map {
        key: "peptides_in:0"
        value: "peptides_in:0"
      }
    },
    {
      model_name: "Prosit_2019_intensity"
      model_version: 1
      input_map {
        key: "peptides_in:0"
        value: "peptides_in:0"
      },
      input_map {
        key: "collision_energy_in:0"
        value: "collision_energy_in:0"
      },
      input_map {
        key: "precursor_charge_in:0"
        value: "precursor_charge_in_preprocessed:0"
      }
      output_map {
        key: "out/Reshape:0"
        value: "out/Reshape:0"
      }
    },
    {
      model_name: "Prosit_2019_intensity_postprocess"
      model_version: 1
      input_map {
        key: "peptides_in:0"
        value: "peptides_in_str:0"
      },
      input_map{
        key: "precursor_charge_in:0"
        value: "precursor_charge_in_preprocessed:0"
      }
      input_map{
        key: "peaks_in:0",
        value: "out/Reshape:0"
      }
      output_map {
        key: "out/Reshape:1"
        value: "out/Reshape:1"
      }
    }
  ]
}
```

# Business logic scripting

The business logic model is simply a python script with the filename of "model.py". In that you need to implement a class called "TritonPythonModel". In general you can only implement the member function called "execute(self,requests) -> response" to be able to maniuplate the input tensors. However, if you would like to do initalize certain data (usually the type of output tensors when it comes to pre/post-processing) you can use "def initialize(self,args) -> None" function. Then if you would like to clean up, you can do so in "def finalize(self)" member function.

for instance this code snippet is part of the Prosit_Preprocess_charge business logic: In initialize member function we defined a variable that holds the type of output tensor which we will use at "execute" function. In execute function we do preprocessing work and output the tensor in a variable "reponse" and return it to the user. In finalize step we just print out the serving is completed to the stdin.

```python
class TritonPythonModel:
   def initialize(self,args):
     print("Preprocessing of the Precursor_Charge_In")
     self.model_config = model_config = json.loads(args['model_config'])
     output0_config = pb_utils.get_output_config_by_name(
             self.model_config, "precursor_charge_in:0")
     self.output_dtype = pb_utils.triton_string_to_numpy(
                          output0_config['data_type'])
   def execute(self, requests):
     peptide_in_str = []
     responses = []
     print("Pre-processing of charge is called")
     for request in requests:
      charge_in_raw = pb_utils.get_input_tensor_by_name(request, "precursor_charge_in_int:0")
      charge_in_flat = sum(charge_in_raw.as_numpy().tolist(), [])
      charge_in = to_on_hot(charge_in_flat)
      t = pb_utils.Tensor("precursor_charge_in:0",charge_in.astype(self.output_dtype))
      responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
     return responses
   def finalize(self):
     print('done processing Preprocess charge')
```