import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def tf_nan2num(w):
    return w
    NUMBER = 10e-9
    w = tf.where(tf.math.is_nan(w), tf.ones_like(w) * NUMBER, w);
    return w

def make_gradcam_heatmap_yolo(img_array, model, last_conv_layer_name, idx=10):
    
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    img_array = img_array[np.newaxis,:,:,:]/255.0
    
    with tf.GradientTape() as tape:
        #import pdb;pdb.set_trace()
        last_conv_layer_output, preds = grad_model(img_array)
        pred_1 = tf_nan2num(preds[0][0,:,:,:,4] * preds[0][0,:,:,:,idx+5]) 
        pred_2 = tf_nan2num(preds[1][0,:,:,:,4] * preds[1][0,:,:,:,idx+5])
        
        pred = tf.reduce_sum(tf.maximum(pred_1  , 0.0)) + tf.reduce_sum(tf.maximum(pred_2  , 0.0))
        #print(pred)
        '''
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in preds[:2]]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        valid_scale=[0, np.inf]
        #pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]
         
        #classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[:, idx]
        pred = tf.maximum(scores , 0.0)
        print(tf.reduce_sum(pred))
        '''
    
    grads = tape.gradient(pred, last_conv_layer_output).numpy()
    grads = np.nan_to_num(grads)
    #import pdb;pdb.set_trace()
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0,:,:,:]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    #heatmap = heatmap / tf.math.reduce_max(heatmap)
    return heatmap.numpy(),pooled_grads.numpy(),last_conv_layer_output.numpy()#,grads.numpy()


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, idx=10):
    
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    img_array = img_array[np.newaxis,:,:,:]/255.0
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        preds = tf_nan2num(preds[0][idx]) 
        
    grads = tape.gradient(preds, last_conv_layer_output).numpy()
    grads = np.nan_to_num(grads)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0,:,:,:]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy(),pooled_grads.numpy(),last_conv_layer_output.numpy()#,grads.numpy()

def make_preds_heatmap(img_array, model, last_conv_layer_name, idx=10):
    
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    img_array = img_array[np.newaxis,:,:,:]/255.0
    
    with tf.GradientTape() as tape:
        last_conv_layer_output,preds = grad_model(img_array)
        preds = tf.reduce_max(preds) 
    
    grads = tape.gradient(preds, last_conv_layer_output).numpy()
    grads = np.nan_to_num(grads)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0,:,:,:]
    #import pdb;pdb.set_trace()
    heatmap = last_conv_layer_output * pooled_grads#[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    '''
    pooled_grads = tf.reduce_mean(preds, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0,:,:,:]
    heatmap = pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    '''
    #heatmap = heatmap / tf.math.reduce_max(heatmap)
    return heatmap.numpy(),grads,last_conv_layer_output.numpy()#,grads.numpy()




def overlay_on_image(img, heatmap, ax, alpha=1.0):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    ax.imshow(superimposed_img)
