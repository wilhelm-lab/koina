#!/usr/bin/python3
from itertools import zip_longest
import logging
import time
from pathlib import Path
import yaml
import requests
from jinja2 import Environment, FileSystemLoader


def load_yaml(path):
    with open(path, "r", encoding="UTF-8") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)


def generate_example_code(model, grpc_url, code_template):
    """
    Generates the GRPC examples codes based on the notes
    """
    logging.info("Using grpc url:\t %s", grpc_url)
    logging.info("Using template to create python code:\t\t %s", code_template)
    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template(code_template)
    context = model
    context["url"] = grpc_url
    return template.render(context).replace("\n", "\n        ")


def sleep_until_service_starts(http_server):
    serving_started = False
    wait_time = 10

    url = f"{http_server}/v2/health/ready"
    while not serving_started:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code >= 200 and r.status_code <= 299:
                serving_started = True
                logging.info("Serving started continuing the program")
                return
            logging.info("Waiting for serving to start: %s ", url)
            time.sleep(wait_time)
        except requests.exceptions.ConnectionError:
            logging.info("Waiting for serving to start: %s ", url)
            time.sleep(wait_time)


def get_config(http_url, name):
    # TODO throw an error when the an unknown model is requested
    # {'error': "Request for unknown model: 'Deeplc_hela_hf' is not found"}
    url = http_url + f"/v2/models/{name}/config"
    logging.info("Getting config from:\t\t %s", url)
    r = requests.get(url, timeout=1)
    return r.json()


def create_openapi_yaml(models, tmpl_url):
    # Create the Swagger.yaml based on the template

    openapi_template_file = "web/openapi/templates/openapi.yml"
    logging.info("Using template file:\t\t %s", openapi_template_file)

    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template(openapi_template_file)
    context = {"models": models, "tmpl_url": tmpl_url}

    content = template.render(context)
    with open("web/public/openapi.yml", mode="w", encoding="utf-8") as yam:
        yam.write(content)
    logging.info("Finished Generating the Swagger YAML file.")


def main(http_url, grpc_url, tmpl_url):
    model_dict = {x.parent.name: x for x in Path("models").rglob("notes.yaml")}

    sleep_until_service_starts(http_url)

    models = []
    for name, model_path in model_dict.items():
        logging.info("Start working on model:\t %s", name)
        models.append(
            {
                "name": name,
                "note": load_yaml(model_path),
                "config": get_config(http_url, name),
            }
        )
        models[-1]["note"]["description"] = models[-1]["note"]["description"].replace(
            "\n", "<br>"
        )
        try:
            models[-1]["note"]["citation"] = models[-1]["note"]["citation"].replace(
                "\n", "<br>"
            )
        except KeyError:
            logging.warning("Model %s does not contain a citation", name)
        add_np_and_openapi_dtype(models[-1]["note"])
        copy_outputs_to_note(models[-1])
        verify_inputs(models[-1])
        models[-1]["code"] = generate_example_code(
            models[-1],
            grpc_url,
            code_template="web/openapi/templates/code/python_koinapy.txt",
        )
        models[-1]["curl_code"] = generate_example_code(
            models[-1], grpc_url, code_template="web/openapi/templates/code/curl.txt"
        )
        models[-1]["rlang_code"] = generate_example_code(
            models[-1], grpc_url, code_template="web/openapi/templates/code/rlang.txt"
        )
        models[-1]["rlang_koinar_code"] = generate_example_code(
            models[-1],
            grpc_url,
            code_template="web/openapi/templates/code/rlang_koinar.txt",
        )
        models[-1]["java_15_code"] = generate_example_code(
            models[-1],
            grpc_url,
            code_template="web/openapi/templates/code/java_15.txt",
        )
        models[-1]["java_8_code"] = generate_example_code(
            models[-1],
            grpc_url,
            code_template="web/openapi/templates/code/java_8.txt",
        )

    logging.info("Template URL: %s", tmpl_url)
    create_openapi_yaml(models, tmpl_url)


def copy_outputs_to_note(model_dict):
    model_dict["note"]["outputs"] = [o["name"] for o in model_dict["config"]["output"]]


def verify_inputs(model_dict):
    for x, y in zip_longest(
        model_dict["note"]["examples"]["inputs"], model_dict["config"]["input"]
    ):
        try:
            assert x["name"] == y["name"]
            assert x["httpdtype"] == tritondtype_to_httpdtype(y["data_type"])
        except AssertionError as ex:
            raise AssertionError(
                f"Inputs inconsistent for {model_dict['name']} {x} != {y}"
            ) from ex


def httpdtype_to_npdtype(dtype):
    mapping = {
        "FP32": 'np.dtype("float32")',
        "BYTES": 'np.dtype("O")',
        "INT16": 'np.dtype("int16")',
        "INT32": 'np.dtype("int32")',
        "INT64": 'np.dtype("int64")',
    }
    return mapping[dtype]


def httpdtype_to_openapidtype(dtype):
    if dtype == "BYTES":
        return "string"
    return "number"


def tritondtype_to_httpdtype(dtype):
    if dtype == "TYPE_STRING":
        return "BYTES"
    return dtype.replace("TYPE_", "")


def add_np_and_openapi_dtype(model_note):
    for x in model_note["examples"]["inputs"]:
        x["npdtype"] = httpdtype_to_npdtype(x["httpdtype"])
        x["openapidtype"] = httpdtype_to_openapidtype(x["httpdtype"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_yaml("web/openapi/config.yml")
    main(config["HTTP_URL"], config["TMPLT_GRPC_URL"], config["TMPLT_HTTP_URL"])
