from flask import Flask, Response
from flask_cors import CORS
import os
import subprocess
from model import *
from matplotlib import pyplot as plt
import numpy as np
import io
import base64
import json

app = Flask(__name__)
CORS(app)
model,layers = [],[]
img = get_img(file_path='car/1.jpg')

@app.route('/')
def hello_world():
    return 'Hello World ...'

@app.route('/files/')
def files():
    return os.listdir('./data/')

#resnetv2_50
@app.route('/<model_name>/')
def resnetv2_50(model_name):
    global model,layers
    model , layers = get_model(model_name=model_name)
    return layers

@app.route('/layer/<layer_name>/')
def get_layer_output(layer_name):
    #plot_layer_featmap()
    image_buffer = io.BytesIO()
    #plt.plot(np.arange(100))
    print(layer_name)
    plt = plot_layer_fmaps(model,img,layer_name)
    plt.tight_layout()
    plt.savefig(image_buffer, format='png',dpi=100)
    image_buffer.seek(0)
    image_data = base64.b64encode(image_buffer.read()).decode('utf-8')
    return Response(response=json.dumps({'src':image_data,'content_type':'image/png'}),status=200)

#train/1.jpg
@app.route('/image/<_dir>/<_file>')
def get_img_array(_dir,_file):
    global img
    
    img = get_img(f'{_dir}/{_file}')
    image_buffer = cv2.imencode('.jpg',img)[1]
    image_buffer = io.BytesIO(image_buffer)
    #cv2.imwrite(image_buffer,jpg)
    image_buffer.seek(0)
    image_data = base64.b64encode(image_buffer.read()).decode('utf-8')
    return Response(response=json.dumps({'src':image_data,'content_type':'image/jpg'}),status=200)

@app.route('/graph')
def graph():
    return get_graph()
    
    
if __name__ == '__main__':
    app.run(host='127.0.0.1',port=9001,debug=True)
