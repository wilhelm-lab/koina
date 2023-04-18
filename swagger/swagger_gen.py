#!/usr/bin/python3
import logging
import requests
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import argparse
import os
import json
import yaml
import time

nptype_convert = {
    "FP32": "np.float32",
    "BYTES": "np.object_",
    "INT32": "np.int32",
    "INT64": "np.int64",
}


def get_notes(model_name, model_name_path):
    notes = {}
    with open(os.path.join(model_name_path, "notes.yaml"), "r") as yaml_file:
        notes[model_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return notes


def remove_type_prefix(models):
    for model in models:
        for i in range(0, len(model["input"])):
            model["input"][i]["data_type"] = model["input"][i]["data_type"].replace(
                "TYPE_", ""
            )
            if model["input"][i]["data_type"] == "STRING":
                model["input"][i]["data_type"] = "BYTES"

    return models


def generate_grpc_header_boilierplate(grpc_url, model):
    """
    Generates the header boilerplate code:
        - imports of libraries
        - init of the GRPC server address
        - init of variables
    """
    model_name = model["name"]
    python_code = "<pre>"
    python_code += "import numpy as np<br>"
    python_code += "import time<br>"
    python_code += "import tritonclient.grpc as grpcclient<br>"
    python_code += "outputs=[]<br>"

    code = """server_url = \"{server_url}\"<br>model_name = \"{model_name}\"<br>"""
    python_code += code.format(server_url=grpc_url, model_name=model_name)

    for i in range(0, len(model["output"])):
        model_output = model["output"][i]
        index = str(i + 1)
        model_output_name = model_output["name"]
        out_layer = """out_layer{index}=\"{model_output_name}\" <br>""".format(
            index=index, model_output_name=model_output_name
        )
        python_code += out_layer

    python_code += (
        "triton_client = grpcclient.InferenceServerClient(url=server_url)<br>"
    )

    return python_code


def generate_output_boilerplate(model):
    """
    Generates print result of output layer boilerplate
    """
    python_code = 'print("Result")<br>'
    for i in range(0, len(model["output"])):
        code = """print(np.round(result.as_numpy(out_layer{index}), 1))<br>"""
        code = code.format(index=i + 1)
        python_code += code
    python_code += "</pre>"
    return python_code


def generate_append_output_boilerplate(model):
    """
    Generates output list append of output tensors boilerplate
    """
    python_code = ""
    for i in range(0, len(model["output"])):
        code = """outputs.append(grpcclient.InferRequestedOutput(out_layer{index}))<br>""".format(
            index=i + 1
        )
        python_code += code
    python_code += (
        "result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)<br>"
    )
    return python_code


def generate_input_create_boilerplate(model):
    """
    Generates the GRPC input list boilerplate code
    """
    python_code = """inputs = []<br>"""
    logging.info("input:")
    logging.info(model["note"]["examples"]["input"])
    input_examples = ""
    try:
        input_examples = json.loads(model["note"]["examples"]["input"])
    except Exception as e:
        logging.info(e)
        logging.info(model["note"]["examples"]["input"])
        exit(1)
    for i in range(0, len(input_examples)):
        input_example = input_examples[i]
        input_name = input_example["name"]
        input_shape = input_example["shape"]
        input_data_type = input_example["datatype"]
        code = """inputs.append(grpcclient.InferInput(\"{input_name}\", {shape}, \"{type}\")) <br>""".format(
            input_name=input_name, shape=input_shape, type=input_data_type
        )
        python_code += code

    for i in range(0, len(input_examples)):
        input_example = input_examples[i]
        input_name = input_example["name"]
        input_shape = input_example["shape"]
        input_data_type = input_example["datatype"]
        input_data = input_example["data"]
        logging.info("datatype: ")
        logging.info(input_data_type)
        logging.info(type(input_data_type))
        code = ""
        code = """input{index} = np.array({example},dtype={type}) <br>""".format(
            index=i,
            example=input_data,
            shape=input_shape[1],
            type=nptype_convert[input_data_type],
        )
        python_code += code

    for i in range(0, len(input_examples)):
        code = """inputs[{index}].set_data_from_numpy(input{index}) <br>""".format(
            index=i
        )
        python_code += code
    return python_code


def generate_example_code(model, grpc_url):
    """
    Generates the GRPC examples codes based on the notes
    """
    python_code = generate_grpc_header_boilierplate(grpc_url, model)
    python_code += generate_input_create_boilerplate(model)
    python_code += generate_append_output_boilerplate(model)
    python_code += """start = time.time()<br>"""
    python_code += """result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)<br>"""
    python_code += """end = time.time()<br>"""
    python_code += generate_output_boilerplate(model)

    return python_code


def sleep_until_service_starts(http_server):
    serving_started = False
    wait_time = 10

    url = f"{http_server}/v2/health/ready"
    while not serving_started:
        logging.info("Waiting for serving to start: " + url)
        try:
            r = requests.get(url)
            if r.status_code >= 200 and r.status_code <= 299:
                serving_started = True
                logging.info("Serving started continuing the program")
            else:
                time.sleep(wait_time)
        except Exception as e:
            time.sleep(wait_time)


def main(http_url, tmpl_url, grpc_url):
    logging.basicConfig(level=logging.INFO)
    with open("./swagger/model_names.json", "r") as f:
        model_names = json.load(f)
    triton_url_template = f"{tmpl_url}/v2/models/{{model}}/config"

    # there is a slight delay before service turns healthy
    # therefore sleep just a few seconds
    sleep_until_service_starts(http_url)

    models = []
    for model in model_names:
        key, _ = list(model.items())[0]
        url = http_url + f"/v2/models/{key}/config"
        logging.info("The selected triton back-end to generate swagger from: " + url)
        r = requests.get(url)
        models.append(r.json())
    logging.info("Models: ")
    logging.info(models)

    # Remove the type prefix because the python code doesn't use the same type notations
    models = remove_type_prefix(models)
    for model in model_names:
        name, model_path = list(model.items())[0]
        notes = get_notes(name, model_path)
        for i in range(0, len(models)):
            if models[i]["name"] == name:
                models[i]["note"] = notes[name]
                code = generate_example_code(models[i], grpc_url)
                models[i]["note"]["code"] = code

    # Create the Swagger.yaml based on the template
    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template("swagger/swagger_tmpl.yml")
    context = {"models": models, "tmpl_url": tmpl_url}

    content = template.render(context)
    logging.info("Generating the Swagger YAML file ... ")
    with open("swagger/swagger.yml", mode="w", encoding="utf-8") as yaml:
        yaml.write(content)
    logging.info("Finished Generating the Swagger YAML file ... ")


if __name__ == "__main__":
    http_url = os.getenv("HTTP_URL")
    tmpl_url = os.getenv("TMPL_URL")
    grpc_url = os.getenv("GRPC_URL")
    main(http_url, tmpl_url, grpc_url)
