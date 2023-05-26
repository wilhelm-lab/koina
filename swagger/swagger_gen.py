#!/usr/bin/python3
import logging
import os
import json
import time
from pathlib import Path
import html
import yaml
import requests
from jinja2 import Environment, FileSystemLoader

nptype_convert = {
    "FP32": "np.float32",
    "BYTES": "np.object_",
    "INT32": "np.int32",
    "INT64": "np.int64",
}


def get_notes(model_name, path):
    notes = {}
    with open(path, "r", encoding="UTF-8") as yaml_file:
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


def generate_example_code(model, grpc_url):
    """
    Generates the GRPC examples codes based on the notes
    """
    logging.info("@" * 50)
    logging.info(model)
    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template("swagger/tmpl_python_code.txt")
    context = model
    context["url"] = grpc_url

    txt = html.escape(template.render(context)).replace("\n", "<br>")
    return f"<pre>{txt}</pre>"


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
            else:
                logging.info(f"Waiting for serving to start: {url}")
                time.sleep(wait_time)
        except requests.exceptions.ConnectionError:
            logging.info(f"Waiting for serving to start: {url}")
            time.sleep(wait_time)


def get_configs(names):
    configs = []
    for name in names:
        url = http_url + f"/v2/models/{name}/config"
        logging.info(f"The selected triton back-end to generate swagger from: {url}")
        r = requests.get(url, timeout=1)
        configs.append(r.json())
    return configs


def create_swagger_yaml(models):
    # Create the Swagger.yaml based on the template
    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template("swagger/swagger_tmpl.yml")
    context = {"models": models, "tmpl_url": tmpl_url}

    content = template.render(context)
    logging.info("Generating the Swagger YAML file ... ")
    with open("swagger/swagger.yml", mode="w", encoding="utf-8") as yam:
        yam.write(content)
    logging.info("Finished Generating the Swagger YAML file ... ")


def main(http_url, grpc_url):
    logging.basicConfig(level=logging.INFO)

    model_dict = {x.parent.name: x for x in Path("models").rglob("notes.yaml")}

    # there is a slight delay before service turns healthy
    # therefore sleep just a few seconds
    sleep_until_service_starts(http_url)

    models = get_configs(model_dict.keys())
    logging.info(f"Models: {models}")

    # Remove the type prefix because the python code doesn't use the same type notations
    models = remove_type_prefix(models)
    for name, model_path in model_dict.items():
        notes = get_notes(name, model_path)
        for i, _ in enumerate(models):
            if models[i]["name"] == name:
                models[i]["note"] = notes[name]
                code = generate_example_code(models[i], grpc_url)
                models[i]["note"]["code"] = code

    create_swagger_yaml(models)


if __name__ == "__main__":
    http_url = os.getenv("HTTP_URL")
    tmpl_url = os.getenv("TMPL_URL")
    grpc_url = os.getenv("GRPC_URL")
    main(http_url, grpc_url)
