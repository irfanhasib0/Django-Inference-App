import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from gradcam import *
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