FROM python:3.8
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install libgl1-mesa-glx
COPY ./registry.py /usr/local/lib/python3.8/site-packages/django/apps/
COPY . /code/
EXPOSE 9002

