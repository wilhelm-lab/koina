#!/bin/sh
pipx ensurepath
source ~/.bashrc
poetry install -C./clients/python/ --with develop 
cd web
npm install
touch /tmp/done_setup # For healthcheck
sleep infinity
