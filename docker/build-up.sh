#!/usr/bin/env bash
# build
docker build -t cuda-python:latest .
# up
docker run -d -it --rm -v $(PWD)/../bayopt:/var/project/bayopt --name cuda-python_1 cuda-python:latest /bin/bash