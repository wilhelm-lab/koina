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

    return models
def generate_intensity_python_code(model,server_url):
    model_name = model['name']
    python_code = '''
    <pre>
    import numpy as np <br>
    import numpy as np <br>
    import time <br>
    import tritonclient.grpc as grpcclient <br>
    if __name__ == \"__main__\" <br>
    <blockquote> server_url = \"{server_url}\"<br>
    model_name = \"{model_name}\"<br>
    '''
    python_code = python_code.format(server_url=server_url,model_name=model_name)
    for i in range(0,len(model['output'])):
        model_output = model['output'][i]
        index=str(i)
        model_output_name = model_output['name']
        out_layer='''out_layer{index}=\"{model_output_name}\"<br>'''.format(index=index,model_output_name=model_output_name)
        python_code+=out_layer

    python_code+='''
    batch_size = 5<br>
    inputs = []<br>
    triton_client = grpcclient.InferenceServerClient(url=server_url)<br>
    '''
    
    for i in range(0,len(model['input'])):
        data_type = model['input'][i]['data_type']
        input_tensor = model['input'][i]['name']
        input_code='''inputs.append(grpcclient.InferInput(\"{input_tensor}\",[batch_size, 1],{data_type}))<br>'''.format(input_tensor=input_tensor,data_type=data_type)
        python_code+=input_code
    python_code+="# Create the data for the two input tensors. Initialize the first \ n# to unique integers and the second to all ones. <br>" 

    python_code+= '''    
    peptide_seq_in = np.array([[\"AAAAAKAKM[UNIMOD:35]\"] for i in range(0, batch_size)], dtype=np.object_)<br>"
    ce_in = np.array([[25] for i in range(0, batch_size)], dtype=np.float32)<br>
    precursor_charge_in = np.array([[2] for i in range(0, batch_size)], dtype=np.int32)<br>
    print("len: " + str(len(inputs)))<br>
    inputs[0].set_data_from_numpy(peptide_seq_in)<br>
    inputs[1].set_data_from_numpy(ce_in)<br>
    inputs[2].set_data_from_numpy(precursor_charge_in)<br>
    '''
    out="outputs = ["
    for i in range(0,len(model['output'])):
        outputs='''grpcclient.InferRequestedOutput(out_layer{index}), '''.format(index=i)
        out+=outputs
    out+="]<br>"
    python_code+=out
    python_code+='''start = time.time()<br>
    result = triton_client.infer(model_name, inputs=inputs, outputs=outputs)<br>
    end = time.time()<br>
    print(\"Time: \" + str(end - start))<br>
    print("Result")<br>
    print(np.round(result.as_numpy(out_layer1), 1))<br>
    print(np.round(result.as_numpy(out_layer2), 1))<br>
    \t  </pre>
    '''
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
    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template("swagger_tmpl.yml")

    for model in model_names:
        name,model_path = list(model.items())[0]
        notes = get_notes(name,model_path)
        for i in range(0,len(models)):
            if (models[i]['name'] == name):
                models[i]['note'] = notes[name]
                if ("intensity" in name):
                    code = generate_intensity_python_code(models[i],reverse_host+":"+reverse_port)
                    models[i]['note']['code'] = code

    context = {'models': models,'triton_server_ip':reverse_host,
              'triton_server_port':reverse_port}

    content = template.render(context)
    logging.info("Generating the Swagger YAML file ... ")
    with open("swagger.yml", mode="w", encoding="utf-8") as yaml:
        yaml.write(content)
    logging.info("Finished Generating the Swagger YAML file ... ")


if __name__ == "__main__":
    main()
