#FROM tensorflow/tensorflow:2.3.0-gpu-jupyter 
FROM nvidia/cuda:11.6.0-base-ubuntu20.04
WORKDIR /code
COPY requirements.txt /code/
RUN apt update -y 
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.8 python3-dev python3-pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get install wget
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
#RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install libgl1-mesa-glx
COPY . /code/
