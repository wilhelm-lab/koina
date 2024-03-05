# Koina

## Accessing a public server
### cURL
Here is an example http request using only cURL sending a POST request to with a json body.

```bash
curl "https://koina.wilhelmlab.org/v2/models/Prosit_2019_intensity/infer" \
 --data-raw '
{
  "id": "LGGNEQVTR_GAGSSEPVTGLDAK",
  "inputs": [
    {"name": "peptide_sequences",   "shape": [2,1], "datatype": "BYTES", "data": ["LGGNEQVTR","GAGSSEPVTGLDAK"]},
    {"name": "collision_energies",  "shape": [2,1], "datatype": "FP32",  "data": [25,25]},
    {"name": "precursor_charges",    "shape": [2,1], "datatype": "INT32", "data": [1,2]}
  ]
}'
```
### Python
For examples of how to access models using python you can check out [our OpenAPI documentation ](https://koina.wilhelmlab.org/docs/#overview).

### R
TODO

## Hosting your own server

### Dependencies
Koina depends on [docker](https://docs.docker.com/engine/install/) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

You can find an ansible script that installs all dependencies [here](docs/server/).

### How to run it
After installing the dependencies you can pull the docker image and run it. If you have multiple GPUs installed on your server you can choose which one is used by modifying `--gpus '"device=0"'`
When using this docker image you need to accept the terms in the [NVIDIA Deep Learning Container License](NVIDIA_Deep_Learning_Container_License.pdf)
```bash
docker run \
    --gpus '"device=0" \
    --shm-size 8G \
    --name koina \
    -p 8500-8502:8500-8502 \
    -d \
    --restart unless-stopped \
    ghcr.io/wilhelm-lab/koina:latest
```

If you want to stay up to date with the latest version of Koina we suggest you also deploy containrrr/watchtower.


```bash
docker run \
  -d \
  --name watchtower \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --restart unless-stopped \
  containrrr/watchtower -i 30
```

## Adding your own model

### Set up a development server

1. Install dependencies ([Ansible script](docs/server/))
2. (Suggested) Install [docker compose](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
3. Clone the repo
5. Update `.env` with your user- and group-id to avoid file permission issues
6. Start the server with `docker compose up -d --wait`
7. Confirm that the server started successfully with `docker compose logs -f serving`. It the startup wass successful you will see something like this.:
```
koina-serving-1  | I0615 13:27:04.260871 90 grpc_server.cc:2450] Started GRPCInferenceService at 0.0.0.0:8500
koina-serving-1  | I0615 13:27:04.261163 90 http_server.cc:3555] Started HTTPService at 0.0.0.0:8501
koina-serving-1  | I0615 13:27:04.303178 90 http_server.cc:185] Started Metrics Service at 0.0.0.0:8502
```

Further considerations
- For development we suggest to use Visual Studio Code with the `Dev Containers` and `Remote - SSH` extensions.
  Using this system you can connect to the server and open the cloned git repo. You will be prompted to reopen the folder in a DevContainer where a lot of useful dependencies are already installed including the dependencies required for testing, linting and styling. Using the dev-container you can lint your code by running `lint`, run tests with `pytest` and style your code with `black .`
- From within the dev-container you can get requests from the `serving` container by providing the url `serving:8501` for http and `serving:8501` for gRPC.

### Import model files
Triton supports all major machine learning frameworks. The format you need to save your model in depends on the framework used to train your model. For detailed instructions you can check out this [documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md#model-files).
You can find examples for [TensorFlow](models/Prosit/Prosit_2019_intensity/1), [PyTorch](models/AlphaPept/AlphaPept_ms2_generic/1) and [XGBoost](models/ms2pip/model_20210416_HCD2021_Y/1) in our model repository.

#### Model repository
For storing the model files themselves we use Zenodo. If you want to add your model to the publicly available Koina instances, You should upload your model file to Zenodo and commit a file named `.zenodo` containing the download url in place of the real model file.

### Create pre- and post-processing steps
A major aspect of Koina, is that all models share a common interface making it easier for clients to use all models.
Triton supports models written in pure python. If your model requires pre- and/or post-processing you can implement this as a "standalone" model in python.
There are numerous examples in this repository. One with low complexity you can find [here](models/AlphaPept/AlphaPept_Preprocess_charge/1).
If you made changes to your model you need to restart Triton. You can do that with `docker compose restart serving`.

### Create an ensemble model to connect everything
The pre- and postprocessing models you just implemented need to be connected to the
Ensemble models don't have any code themselves they just manage moving tensors between other models. This is perfect for combining your potentially various pre- and post-processing steps with your main model to create one single model/workflow.

### Write tests for your model
To make sure that your model was implemented correctly and future changes do not make any unforseen changes you can add tests for it in the `test` folder. The files added there should match the model name used in the model repository.
