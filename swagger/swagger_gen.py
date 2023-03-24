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
grpc_url= os.getenv("TRITON_SERVER_IP")+":8504"

def get_notes(model_name,model_name_path):
    notes = {}
    with open(os.path.join(model_name_path,"notes.yaml"),'r') as yaml_file:
        notes[model_name] = yaml.load(yaml_file,Loader=yaml.FullLoader)
    return notes

def remove_type_prefix(models):
    for model in models:
        for i in range(0, len(model["input"])):
            model["input"][i]["data_type"] = model["input"][i]["data_type"].replace(
                "TYPE_", ""
            )
            if (model["input"][i]["data_type"] == "STRING"):
                model["input"][i]["data_type"] = "BYTES"

    return models

def generate_grpc_header_boilierplate(model):
    model_name = model['name']
    python_code = '''
    <pre>
    import numpy as np<br>
    import time<br>
    import tritonclient.grpc as grpcclient<br>
    outputs=[]<br>
    server_url = \"{server_url}\"<br>
    model_name = \"{model_name}\"<br>
    '''
    python_code = python_code.format(server_url=grpc_url,model_name=model_name)

    for i in range(0,len(model['output'])):
        model_output = model['output'][i]
        index=str(i+1)
        model_output_name = model_output['name']
        out_layer='''out_layer{index}=\"{model_output_name}\" <br>'''.format(index=index,model_output_name=model_output_name)
        python_code+=out_layer

    python_code+='''
    triton_client = grpcclient.InferenceServerClient(url=server_url)<br>
    '''

    return python_code

def generate_output_boilerplate(model):
    python_code='''print("Result")<br>'''
    for i in range(0,len(model['output'])):
        code='''print(np.round(result.as_numpy(out_layer{index}), 1))<br>'''
        code = code.format(index=i+1)
        python_code+=code
    python_code+="</pre>"
    return python_code

def generate_append_output_boilerplate(model):
    python_code=''
    for i in range(0,len(model['output'])):
        code='''
            outputs.append(grpcclient.InferRequestedOutput(out_layer{index}))<br>
        '''.format(index=i+1)
        python_code+=code
    python_code+="result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)<br>"
    return python_code

def generate_intensity_python_code(model):
    python_code = generate_grpc_header_boilierplate(model)
    python_code += '''
    batch_size = 5<br>
    inputs = []<br>
    inputs.append(grpcclient.InferInput("peptides_in_str:0", [batch_size, 1], "BYTES"))<br>
    inputs.append(<br>
        grpcclient.InferInput("collision_energy_in:0", [batch_size, 1], "FP32")<br>
    )<br>
    inputs.append(<br>
        grpcclient.InferInput("precursor_charge_in_int:0", [batch_size, 1], "INT32")<br>
    )<br>
    peptide_seq_in = np.array([["AAAAAKAKM[UNIMOD:35]"] for i in range(0, batch_size)], dtype=np.object_)<br>
    ce_in = np.array([[25] for i in range(0, batch_size)], dtype=np.float32)<br>
    precursor_charge_in = np.array([[2] for i in range(0, batch_size)], dtype=np.int32)<br>
    print("len: " + str(len(inputs)))<br>
    inputs[0].set_data_from_numpy(peptide_seq_in)<br>
    inputs[1].set_data_from_numpy(ce_in)<br>
    inputs[2].set_data_from_numpy(precursor_charge_in)<br>
    '''
    python_code+=generate_append_output_boilerplate(model)
    '''
    start = time.time()<br>
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)<br>
    end = time.time()<br>
    '''
    python_code+=generate_output_boilerplate(model)

    return python_code
   

def generatems2pip_pythong_code(model):
    python_code = generate_grpc_header_boilierplate(model)
    python_code+='''
    peptides = [["ACDEK/2"], ["AAAAAAAAAAAAA/3"]]<br>
    batch_size = len(peptides)<br>
    inputs = []<br>
    outputs = []<br>
    triton_client = grpcclient.InferenceServerClient(url=server_url)<br>
    inputs.append(grpcclient.InferInput("proforma_ensemble", [batch_size, 1], "BYTES"))<br>
    peptide_seq_in = np.array([i for i in peptides], dtype=np.object_)<br>
    print("len: " + str(len(inputs)))<br>
    inputs[0].set_data_from_numpy(peptide_seq_in)<br>
    '''
    python_code+=generate_append_output_boilerplate(model)
    '''
    start = time.time()<br>
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)<br>
    end = time.time()<br>
    print("Time: " + str(end - start))<br>
    print("Result")<br>
    print(result.as_numpy(out_layer))<br>
    tt = result.as_numpy(out_layer)<br>
    print(tt.reshape((-1, 29)))<br>
    '''
    python_code+=generate_output_boilerplate(model)
    return python_code

def main():
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    with open('./model_names.json','r') as f:
        model_names = json.load(f)
    print(model_names[0])
    serving_started = False
    wait_time = 10
    reverse_host = os.getenv("TRITON_SERVER_IP")
    reverse_port = os.getenv("TRITON_REVERSE_PROXY_SERVER_PORT")
    triton_url_template = "http://" +reverse_host+ ":" + reverse_port + "/v2/models/{model}/config"

    while (not serving_started):
        key,val = list(model_names[0].items())[0]
        url = triton_url_template.format(model=key)
        logging.info("Waiting for serving to start: " + url)
        try:
            r = requests.get(url)
            if (r.status_code >= 200 and r.status_code <= 299):
                serving_started = True
                logging.info("Serving started continuing the program")
            else:
                time.sleep(wait_time)
        except Exception as e:
            time.sleep(wait_time)

    models = []
    for model in model_names:
        key,_ = list(model.items())[0]
        url = triton_url_template.format(model=key)
        logging.info("The selected triton back-end to generate swagger from: " + url)
        r = requests.get(url)
        models.append(r.json())
    logging.info("Models: ")
    logging.info(models)
    # Remove the type prefix because the python code doesn't use the same type notations
    models = remove_type_prefix(models)
    for model in model_names:
        name,model_path = list(model.items())[0]
        notes = get_notes(name,model_path)
        for i in range(0,len(models)):
            if (models[i]['name'] == name):
                models[i]['note'] = notes[name]
                if ("intensity" in name and "prosit" in name.lower()):
                    code = generate_intensity_python_code(models[i])
                    models[i]['note']['code'] = code
                if ("ms2pip" in name ):
                    code = generatems2pip_pythong_code(models[i])
                    models[i]['note']['code'] = code

    # Create the Swagger.yaml based on the template
    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template("swagger_tmpl.yml")
    context = {'models': models,'triton_server_ip':reverse_host,
              'triton_server_port':reverse_port}

    content = template.render(context)
    logging.info("Generating the Swagger YAML file ... ")
    with open("swagger.yml", mode="w", encoding="utf-8") as yaml:
        yaml.write(content)
    logging.info("Finished Generating the Swagger YAML file ... ")


if __name__ == "__main__":
    main()
