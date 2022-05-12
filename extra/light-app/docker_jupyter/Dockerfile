FROM tensorflow/tensorflow:2.3.0-gpu-jupyter 
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY requirements.txt /code/
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt -y install python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt
RUN apt -y install libgl1-mesa-glx
COPY . /code/