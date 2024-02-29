#!/bin/sh
source ~/.bashrc
poetry install --with develop -C ./clients/python/
cd web npm install
sleep infinity
