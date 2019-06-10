#!/usr/bin/env bash
# build
docker build -t cuda-python:latest .
# up
docker run -d -it --rm -v $(PWD)/..:/var/project/bayopt --name bayopt cuda-python:latest /bin/bash