#!/bin/bash
source /home/devuser/.bashrc
cd /workspace/koina/
python ./openapi/openapi_gen.py
cd web
npm run generate
