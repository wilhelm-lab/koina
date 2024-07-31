#!/bin/bash
source ~/.bashrc
cd /workspace/koina/
python ./web/openapi/openapi_gen.py
cd web
npm run generate
