#!/bin/sh
pipx ensurepath
source ~/.bashrc
poetry install --with develop -C ./clients/python/
cd web
npm install
touch /tmp/done_setup # For healthcheck
sleep infinity
