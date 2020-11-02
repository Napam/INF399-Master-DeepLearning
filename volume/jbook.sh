#!/usr/bin/env bash

mkdir -p $HOME/.jupyter/custom
cp /app/.jupyter/custom/custom.css $HOME/.jupyter/custom/

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --config='.jupyter/jupyter_notebook_config.py' --allow-root
