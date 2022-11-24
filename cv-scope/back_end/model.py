import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from gradcam import *
from visualize import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D,\
                                    Cropping2D, UpSampling2D, Add, Softmax, Conv2DTranspose, concatenate,\
                                    GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, ReLU
from tensorflow.keras.activations import sigmoid as Sigmoid
from tensorflow.keras.regularizers import l2

tf.random.set_seed(0)
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
def get_model(model_name='resnet_v2_50'):
    if model_name == 'resnetv2_50':
        from applications.resnet_v2 import ResNet50V2
        model = ResNet50V2(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=(256,256,3),
                     pooling=None,
                     classes=1000)
        layers = [layer.name for layer in model.layers]
        
    if model_name == 'mobilenetv2':
        from applications.mobilenet_v2 import MobileNetV2
        model = MobileNetV2(include_top=True,
                     alpha   = 1.0,
                     weights ='imagenet',
                     input_tensor = None,
                     input_shape  = (256,256,3),
                     pooling = None,
                     classes = 1000)
        layers = [layer.name  for layer in model.layers]
        
    for layer in model.layers:
        shapes=[]
        for var in layer.trainable_variables:
            shapes += [var.numpy().shape]

        if len(shapes): print(layer.name,'\n',shapes)
        else: print(layer.name)
        
    return model,layers

def b2r(x):
    return cv2.cvtColor(x,cv2.COLOR_BGR2RGB)

def get_img(file_path='train/1.jpg'):
    path = f'./data/imgs/{file_path}' 
    img = cv2.imread(path)
    img = b2r(img)
    img = cv2.resize(img,(256,256))
    return img

def plot_fmap(_arr):
    N = _arr.shape[-1]
    m = int(np.sqrt(N))
    n = N//m
    feat_size = _arr.shape[0]
    temp = np.zeros((feat_size*m-1,feat_size*n-1))
    for i in range(0,m-1):
        for j in range(0,n-1):
            #print(i,j,i*m+j)
            temp[i*feat_size:(i+1)*feat_size,j*feat_size:(j+1)*feat_size] = _arr[:,:,i*m+j]#/arr[:,:,i*m+j].max()
    fig = plt.figure(figsize=(10,10))
    plt.imshow(temp)
    plt.xticks(range(0,feat_size*m-1,feat_size),rotation=90)
    plt.yticks(range(0,feat_size*m-1,feat_size),rotation=0)
    plt.grid(linewidth=0.3)
    return plt
    
lyrs = ['conv2_block3_out','conv3_block3_out','conv4_block3_out','conv5_block3_out']
def plot_layer_fmaps(model,img,conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(conv_layer_name).output, model.output])
    img_array  = img[np.newaxis,:,:,:]/255.0
    arr,preds  = grad_model(img_array)
    return plot_fmap(arr[0])   


def GenerateGraph(g, opcode_mapper):
  """Produces the HTML required to have a d3 visualization of the dag."""

  def TensorName(idx):
    return "t%d" % idx

  def OpName(idx):
    return "o%d" % idx

  edges = []
  nodes = []
  first = {}
  second = {}
  pixel_mult = 200  # TODO(aselle): multiplier for initial placement
  width_mult = 170  # TODO(aselle): multiplier for initial placement
  for op_index, op in enumerate(g["operators"] or []):
    if op["inputs"] is not None:
      for tensor_input_position, tensor_index in enumerate(op["inputs"]):
        if tensor_index not in first:
          first[tensor_index] = ((op_index - 0.5 + 1) * pixel_mult,
                                 (tensor_input_position + 1) * width_mult)
        edges.append({
            "source": TensorName(tensor_index),
            "target": OpName(op_index)
        })
    if op["outputs"] is not None:
      for tensor_output_position, tensor_index in enumerate(op["outputs"]):
        if tensor_index not in second:
          second[tensor_index] = ((op_index + 0.5 + 1) * pixel_mult,
                                  (tensor_output_position + 1) * width_mult)
        edges.append({
            "target": TensorName(tensor_index),
            "source": OpName(op_index)
        })

    nodes.append({
        "id": OpName(op_index),
        "name": opcode_mapper(op["opcode_index"]),
        "group": 2,
        "x": pixel_mult,
        "y": (op_index + 1) * pixel_mult
    })
  for tensor_index, tensor in enumerate(g["tensors"]):
    initial_y = (
        first[tensor_index] if tensor_index in first else
        second[tensor_index] if tensor_index in second else (0, 0))

    nodes.append({
        "id": TensorName(tensor_index),
        "name": "%r (%d)" % (getattr(tensor, "shape", []), tensor_index),
        "group": 1,
        "x": initial_y[1],
        "y": initial_y[0]
    })
  graph_str = json.dumps({"nodes": nodes, "edges": edges})

  return graph_str

def get_graph():
    with open('./data/resnetv2_50.tflite', "rb") as file_handle:
        file_data = bytearray(file_handle.read())
        data = CreateDictFromFlatbuffer(file_data)

    opcode_mapper = OpCodeMapper(data)
    g = (data["subgraphs"])[0]
    graph = GenerateGraph(g, opcode_mapper)
    return graph
    