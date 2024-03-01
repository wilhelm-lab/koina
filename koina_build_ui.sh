#!/bin/bash
cd /workspace/koina/web
python ./openapi/openapi_gen.py
npm run generate
