#!/bin/bash
python web/openapi/openapi_gen.py
cd web
npm run generate