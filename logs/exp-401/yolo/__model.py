from .config import *
from .tf import *
import numpy as np
np.random.seed(SEED)

STRIDES         = np.array(YOLO_STRIDES)
ANCHORS         = (np.array(YOLO_ANCHORS).T/STRIDES).T

class BatchNormalization(BatchNormalization):
        # "Frozen state" and "inference mode" are two separate concepts.
        # `layer.trainable = False` is to freeze the layer, so the layer will use
        # stored moving `var` and `mean` in the "inference mode", and both `gama`
        # and `beta` will not be updated !
        def call(self, x, training=False):
            if not training:
                training = tf.constant(False)
            training = tf.logical_and(training, self.trainable)
            return super().call(x, training)

class YoloModel():
    def __init__(self,training=False):
        self.training=training
        self.N = int(16 / TRAIN_MODEL_SCALE) 
        self.yolo_model=self.load_yolo_model()
        
    def get_model(self):
        return self.yolo_model
    
    def convolutional(self,input_layer, filters_shape, downsample=False, activate=True, bn=True):
        if downsample:
            input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                      padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.constant_initializer(0.))(input_layer)
        if bn:
            conv = BatchNormalization()(conv)
        if activate == True:
            conv = LeakyReLU(alpha=0.1)(conv)

        return conv
    
    
    def residual_block(self,input_layer, input_channel, filter_num1, filter_num2):
        short_cut = input_layer
        conv = self.convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
        conv = self.convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2))

        residual_output = short_cut + conv
        return residual_output
    
    def squeeze_excite_block(tensor, ratio=16):
        init = tensor
        channel_axis = 1 if tf.image_data_format() == "channels_first" else -1
        filters = init._keras_shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        if K.image_data_format() == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = multiply([init, se])
        return x

    def upsample(self,input_layer):
        return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

    
    def darknet53(self,input_data):
        input_data = self.convolutional(input_data, (3, 3,  3,  32))
        input_data = self.convolutional(input_data, (3, 3, 32,  64), downsample=True)

        for i in range(1):
            input_data = self.residual_block(input_data,  64,  32, 64)

        input_data = self.convolutional(input_data, (3, 3,  64, 128), downsample=True)

        for i in range(2):
            input_data = self.residual_block(input_data, 128,  64, 128)

        input_data = self.convolutional(input_data, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 256, 128, 256)

        route_1 = input_data
        input_data = self.convolutional(input_data, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 512, 256, 512)

        route_2 = input_data
        input_data = self.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            input_data = self.residual_block(input_data, 1024, 512, 1024)

        return route_1, route_2, input_data
    
    
    def yolov3(self,input_layer, NUM_CLASS):
        # After the input layer enters the Darknet-53 network, we get three branches
        route_1, route_2, conv = self.darknet53(input_layer)
        # See the orange module (DBL) in the figure above, a total of 5 Subconvolution operation
        conv = self.convolutional(conv, (1, 1, 1024,  512))
        conv = self.convolutional(conv, (3, 3,  512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024,  512))
        conv = self.convolutional(conv, (3, 3,  512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024,  512))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 512, 1024))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255] 
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1,  512,  256))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter  
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)
        conv = self.convolutional(conv, (1, 1, 768, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv_mobj_branch = self.convolutional(conv, (3, 3, 256, 512))

        # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv = self.convolutional(conv, (1, 1, 384, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv_sobj_branch = self.convolutional(conv, (3, 3, 128, 256))

        # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255]
        conv_sbbox = self.convolutional(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS +5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]
    
    def yolo_micro(self,input_data):
        input_data = self.convolutional(input_data, (3, 3, 3, self.N))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, self.N, 2*self.N))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 2*self.N, 4*self.N))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 4*self.N, 8*self.N))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 8*self.N, 16*self.N))
        route_1 = input_data
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 16*self.N, 32*self.N))
        input_data = MaxPool2D(2, 1, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 32*self.N, 64*self.N))

        return route_1, input_data
    
    def yolov3_micro(self,input_layer, NUM_CLASS):
        # After the input layer enters the Darknet-19 network, we get 2 branches
        route_1, conv = self.yolo_micro(input_layer)

        conv = self.convolutional(conv, (1, 1, 64*self.N, 16*self.N))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 16*self.N, 32*self.N))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 26, 26, 255]
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 32*self.N, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 16*self.N, 8*self.N))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter  
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv_mobj_branch = self.convolutional(conv, (3, 3, 8*self.N, 16*self.N))
        # conv_mbbox is used to predict medium size objects, shape = [None, 13, 13, 255]
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 16*self.N, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]
    
    def conv_ups_block(self,y1,y2,d,n=2):
                
        y1 = Conv2D(d,(1,1))(y1)
        y1 = UpSampling2D(size=(n,n))(y1)
        y2 = Conv2D(d,(1,1))(y2)
        y = Add()([y1,y2])
        y = BatchNormalization()(y)
        y = LeakyReLU(alpha=0.1)(y)
        #y = UpSampling2D(size=(n,n))(y)
        return y
    
    def Ups2D(self,x):
        x = Conv2DTranspose(self.seg_out_shape, 1,strides=(2,2), activation = 'relu', padding = 'same')(x)
        return x
    
    def yolo_lite(self,input_data):
        x1 = self.convolutional(input_data, (3, 3, 3, self.N))
        x1 = MaxPool2D(2, 2, 'same')(x1)
        route_0 = x1
        x2 = self.convolutional(x1, (3, 3, self.N, 2*self.N))
        x2 = MaxPool2D(2, 2, 'same')(x2)
        x3 = self.convolutional(x2, (3, 3, 2*self.N, 4*self.N))
        route_1 = x3
        x3 = MaxPool2D(2, 2, 'same')(x3)
        x4 = self.convolutional(x3, (3, 3, 4*self.N, 8*self.N))
        route_2 = x4
        x4 = MaxPool2D(2, 2, 'same')(x4)
        x5 = self.convolutional(x4, (3, 3, 8*self.N, 8*self.N))
        route_3 = x5
        x5 = MaxPool2D(2, 2, 'same')(x5)
        x6 = self.convolutional(x5, (3, 3, 8*self.N, 8*self.N))
        x6 = MaxPool2D(2, 1, 'same')(x6)
        x7 = self.convolutional(x6, (3, 3, 8*self.N, 16*self.N))
                
        return route_0, route_1, route_2, route_3 , x7
    
      
        
    def yolov3_lite(self,input_layer, NUM_CLASS):
        # After the input layer enters the Darknet-19 network, we get 2 branches
        route_0, route_1, route_2, route_3 , conv = self.yolo_lite(input_layer)
        
        fpn=[]
        self.seg_out_shape = len(CLASS_NAMES) + 1
        
        if TRAIN_USE_SEG_BIN:
            _route_3 = Conv2D(1,(1,1))(route_3)
            _route_2 = Conv2D(1,(1,1))(route_2)
            _route_1 = Conv2D(1,(1,1))(route_1)
            _route_0 = Conv2D(1,(1,1))(route_0)
            
            y = UpSampling2D(size=(2,2))(_route_3)
            y = Add()([y,_route_2])
            
            y = UpSampling2D(size=(2,2))(y)
            y = Add()([y,_route_1])
            
            y = UpSampling2D(size=(2,2))(y)
            y = Add()([y,_route_0])
            
            y = Sigmoid(y)
            #y = BatchNormalization()(y)
            #y = LeakyReLU(alpha=0.1)(y)
            fpn+= [y]
            
        if TRAIN_USE_SEG:
            y1 = self.conv_ups_block(conv,route_3,self.seg_out_shape,n=2)
            y2 = self.conv_ups_block(y1,route_2,self.seg_out_shape,n=2)
            y3 = self.conv_ups_block(y2,route_1,self.seg_out_shape,n=2)
            y4 = UpSampling2D(size=(2,2))(y3)
            y  = Sigmoid(y4)
            fpn += [y]
            
        conv = self.convolutional(conv, (1, 1, 16*self.N, 8*self.N))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 8*self.N, 16*self.N))
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 16*self.N, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 8*self.N, 4*self.N))
        conv = self.upsample(conv)
        
        #h_min=min(conv.shape[1],route_1.shape[1])
        #w_min=min(conv.shape[2],route_1.shape[2])
        #if h_min or w_min: print("Cropping :",h_min,w_min)
        #conv=Cropping2D(cropping=((conv.shape[1]-h_min,0),(conv.shape[2]-w_min,0)))(conv)
        #route_1=Cropping2D(cropping=((route_1.shape[1]-h_min,0),(route_1.shape[2]-w_min,0)))(route_1)
        
        conv = tf.concat([conv, route_3], axis=-1)
        conv_mobj_branch = self.convolutional(conv, (3, 3, 8*self.N, 16*self.N))
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 16*self.N, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]+fpn
    
    
    def yolov3_tiny(self,input_data):
        input_data = self.convolutional(input_data, (3, 3, 3, 16))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 16, 32))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 32, 64))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 64, 128))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 128, 256))
        route_1 = input_data
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 256, 512))
        input_data = MaxPool2D(2, 1, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 512, 1024))

        return route_1, input_data
    
    
    def yolov3_tiny(self,input_layer, NUM_CLASS):
        # After the input layer enters the Darknet-19 network, we get 2 branches
        route_1, conv = self.yolov3_tiny(input_layer)

        conv = self.convolutional(conv, (1, 1, 1024, 256))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 256, 512))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 26, 26, 255]
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 256, 128))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter  
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv_mobj_branch = self.convolutional(conv, (3, 3, 128, 256))
        # conv_mbbox is used to predict medium size objects, shape = [None, 13, 13, 255]
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]
    
    def decode(self,conv_output, NUM_CLASS, i=0):
        # where i = 0, 1 or 2 to correspond to the three grid scales  
        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position     
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
        conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
        conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box 

        # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
        y = tf.range(output_size, dtype=tf.int32)
        y = tf.expand_dims(y, -1)
        y = tf.tile(y, [1, output_size])
        x = tf.range(output_size,dtype=tf.int32)
        x = tf.expand_dims(x, 0)
        x = tf.tile(x, [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
    
    def create_yolo(self,input_size=416, channels=3, training=False):
        input_layer  = Input([input_size, input_size, channels])

        if TRAIN_YOLO_TINY:
            if YOLO_MODEL == 'LITE':
                conv_tensors = self.yolov3_lite(input_layer, NUM_CLASS)
            if YOLO_MODEL == 'Micro':
                conv_tensors = self.yolov3_micro(input_layer, NUM_CLASS)
        else:
            conv_tensors = self.yolov3(input_layer, NUM_CLASS)

        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors[:2]):
            pred_tensor = self.decode(conv_tensor, NUM_CLASS, i)
            if training: output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)
    
        YoloV3 = tf.keras.Model(input_layer, output_tensors+conv_tensors[2:])
        return YoloV3
    
    def load_yolo_weights(self,model, weights_file):
        tf.keras.backend.clear_session() # used to reset layer names
        # load Darknet original weights to TensorFlow model
        if YOLO_TYPE == "yolov3":
            range1 = 75 if not TRAIN_YOLO_TINY else 13
            range2 = [58, 66, 74] if not TRAIN_YOLO_TINY else [9, 12]
        if YOLO_TYPE == "yolov4":
            range1 = 110 if not TRAIN_YOLO_TINY else 21
            range2 = [93, 101, 109] if not TRAIN_YOLO_TINY else [17, 20]

        with open(weights_file, 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

            j = 0
            for i in range(range1):
                if i > 0:
                    conv_layer_name = 'conv2d_%d' %i
                else:
                    conv_layer_name = 'conv2d'

                if j > 0:
                    bn_layer_name = 'batch_normalization_%d' %j
                else:
                    bn_layer_name = 'batch_normalization'

                conv_layer = model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in range2:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in range2:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

            assert len(wf.read()) == 0, 'failed to read all data'
    
    def load_yolo_model(self):

        if YOLO_FRAMEWORK == "tf": # TensorFlow detection
            if YOLO_TYPE == "yolov4":
                Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
            if YOLO_TYPE == "yolov3":
                Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

            if YOLO_CUSTOM_WEIGHTS == False:
                print("Loading Darknet_weights from:", Darknet_weights)
                yolo = self.create_yolo(input_size=YOLO_INPUT_SIZE,training=self.training)
                if not self.training : self.load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
            else:
                print("Loading custom weights from:", TRAIN_MODEL_PATH)
                yolo = self.create_yolo(input_size=YOLO_INPUT_SIZE,training=self.training)
                if not self.training : yolo.load_weights(TRAIN_MODEL_PATH)  # use custom weights

        elif YOLO_FRAMEWORK == "trt": # TensorRT detection
            saved_model_loaded = tf.saved_model.load(YOLO_CUSTOM_WEIGHTS, tags=[tag_constants.SERVING])
            signature_keys = list(saved_model_loaded.signatures.keys())
            yolo = saved_model_loaded.signatures['serving_default']

        return yolo


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * inter_area / union_area
    iou = tf.maximum(iou,np.finfo(np.float32).eps)
    return iou

def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula  
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

# testing (should be better than giou)
def bbox_ciou(boxes1, boxes2):
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
    up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
    right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
    down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bbox_iou(boxes1, boxes2)

    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
    d = u / c

    ar_gt = boxes2[..., 2] / boxes2[..., 3]
    ar_pred = boxes1[..., 2] / boxes1[..., 3]

    ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term

@tf.function
def calc_yolo_loss(pred, conv, label, bboxes, i=0):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_iou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # Find the value of IoU with the real box The largest prediction box
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < YOLO_IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Calculate the loss of confidence
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_sum = tf.math.reduce_sum(giou_loss, axis=[1,2,3,4])
    conf_sum = tf.math.reduce_sum(conf_loss, axis=[1,2,3,4])
    prob_sum = tf.math.reduce_sum(prob_loss, axis=[1,2,3,4])
    total_loss_sum = giou_sum + conf_sum + prob_sum
    
    giou_loss = tf.math.reduce_mean(giou_sum)
    conf_loss = tf.math.reduce_mean(conf_sum)
    prob_loss = tf.math.reduce_mean(prob_sum)
    
    giou_std = tf.math.reduce_std(giou_sum)
    conf_std = tf.math.reduce_std(conf_sum)
    prob_std = tf.math.reduce_std(prob_sum)
    
    
    _loss_dict={}; _smp_loss_dict={}
    _loss_dict['iou_loss'] = giou_loss#.numpy()
    _loss_dict['conf_loss'] = conf_loss#.numpy() 
    _loss_dict['prob_loss'] = prob_loss#.numpy()
    _loss_dict['det_loss'] = giou_loss + conf_loss + prob_loss #.numpy()

    _loss_dict['iou_std'] = giou_std#.numpy()
    _loss_dict['conf_std'] = conf_std#.numpy() 
    _loss_dict['prob_std'] = prob_std#.numpy()
    _loss_dict['det_std'] = giou_std + conf_std + prob_std #.numpy()
    
    _smp_loss_dict['det'] = total_loss_sum
    return _loss_dict, _smp_loss_dict, giou_loss + conf_loss + prob_loss

#@tf.function
def calc_seg_loss(seg_label,seg_pred):
    
    if TRAIN_USE_SEG_BIN:     
        _label = tf.cast(seg_label!=80,dtype=tf.float32)[:,:,:,tf.newaxis]
        _pred = seg_pred[0][:,:,:,tf.newaxis]
        bce  = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        bce_loss = bce(_label,_pred)
        bce_loss_mean = tf.math.reduce_mean(bce_loss,axis=[1,2])
        
        std  = tf.math.reduce_std(bce_loss_mean)
        loss = tf.math.reduce_mean(bce_loss_mean)
        _loss_dict={}; _smp_loss_dict={}
        _loss_dict['seg_loss']= loss
        _loss_dict['seg_std']=std
        _smp_loss_dict['seg'] = bce_loss_mean
    '''   
    if TRAIN_USE_SEG:     
        cce  = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        cce_loss = cce(seg_label,seg_pred)
        cce_loss_mean = tf.math.reduce_mean(cce_loss,axis=[1,2])
        std  = tf.math.reduce_std(cce_loss_mean)
        loss = tf.math.reduce_mean(cce_loss_mean)
        _loss_dict={}; _smp_loss_dict={}
        _loss_dict['seg_loss']= loss
        _loss_dict['seg_std']=std
        _smp_loss_dict['seg'] = cce_loss_mean
    '''
    
    if TRAIN_USE_SEG:     
        seg_label = tf.one_hot(seg_label,80)
        bce  = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        bce_loss = bce(seg_label,seg_pred)
        bce_loss_mean = tf.math.reduce_mean(bce_loss,axis=[1,2])
        std  = tf.math.reduce_std(bce_loss_mean)
        loss = tf.math.reduce_mean(bce_loss_mean)
        _loss_dict={}; _smp_loss_dict={}
        _loss_dict['seg_loss']= loss
        _loss_dict['seg_std']=std
        _smp_loss_dict['seg'] = bce_loss_mean
        
    
    return _loss_dict, _smp_loss_dict, bce_loss_mean
    
    