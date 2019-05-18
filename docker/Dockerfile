# CUDA Toolkit
# CUDNN
FROM nvidia/cuda:10.1-base as builder

# Update and Setting
RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt update

# Install python3.7
RUN apt install -y python3.7
RUN apt install -y python-pip
# Project directory
ARG project=/var/project

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# Install packages
ADD requirements.txt $project/requirements.txt
RUN pip install -r $project/requirements.txt

WORKDIR $project
