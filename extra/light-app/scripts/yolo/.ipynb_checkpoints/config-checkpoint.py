# YOLO options
TRAIN_CHECKPOINTS_FOLDER    = "logs/exp-4401"
YOLO_TYPE                   = "yolov3" # yolov4 or yolov3
YOLO_MODEL                  = 'LITE'
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"
YOLO_V3_WEIGHTS             = "model_data/yolov3.weights"
YOLO_V4_WEIGHTS             = "model_data/yolov4.weights"
YOLO_V3_TINY_WEIGHTS        = "model_data/yolov3-tiny.weights"
YOLO_V4_TINY_WEIGHTS        = "model_data/yolov4-tiny.weights"
YOLO_TRT_QUANTIZE_MODE      = "INT8" # INT8, FP16, FP32
YOLO_CUSTOM_WEIGHTS         = True
YOLO_COCO_CLASSES           = "model_data/coco/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 256
if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]
if YOLO_TYPE                == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]
# Train options
SEED                        = 0
TRAIN_YOLO_TINY             = True
TRAIN_MODEL_SCALE           = 1
TRAIN_SAVE_BEST_ONLY        = True 
TRAIN_SAVE_WEIGHTS_EVERY    = 1
TRAIN_CLASSES               = "model_data/coco/coco.names"#"model_data/mnist.names"
TRAIN_ANNOT_PATH            = "D:/COCO/annotations_trainval2017/annotations/instances_train2017.txt"
TRAIN_IMG_PATH              = "E:/Datasets/COCO/train2017/"
TRAIN_DATA_SAVE_PATH        = "D:/COCO/train/"
TRAIN_LOGDIR                = "log"
TRAIN_MODEL_PATH            = "logs/exp-401/model_epoch_9_val_det_loss_30.7866/weights"
TRAIN_LOAD_IMAGES_TO_RAM    = False # With True faster training, but need more RAM
TRAIN_BATCH_SIZE            = 64
TRAIN_MINI_BATCH_SIZE       = 16
TRAIN_LR                    = 0.001
TRAIN_INPUT_SIZE            = YOLO_INPUT_SIZE
TRAIN_DATA_AUG              = False
TRAIN_TRANSFER              = True
TRAIN_FROM_CHECKPOINT       = False # "checkpoints/yolov3_custom"
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 2
TRAIN_EPOCHS                = 15

TRAIN_USE_SE_LAYERS         = [0, 0, 0, 0, 0, 0]
TRAIN_USE_DST               = True
TRAIN_USE_SEG               = False
TRAIN_LOSS_WTS              = [1.0, 0.0, 0.0]#1e-8
TRAIN_SEG_BG                = True
TRAIN_SEG_SCALE             = 2
TRAIN_SEG_SUP_CAT           = False

MTL_USE_METHOD              = 'default'
MTL_NO_OF_SHARED_LAYERS     = 12
MTL_CALC_GRAD_VAR           = False
# TEST options
TEST_ANNOT_PATH             = "D:/COCO/annotations_trainval2017/annotations/instances_val2017.txt"
TEST_IMG_PATH               = "E:/Datasets/COCO/val2017/"
TEST_DATA_SAVE_PATH         = "D:/COCO/test/"
TEST_BATCH_SIZE             = 4
TEST_INPUT_SIZE             = YOLO_INPUT_SIZE
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45


#YOLOv3-TINY and YOLOv4-TINY WORKAROUND
if TRAIN_YOLO_TINY:
    YOLO_STRIDES            = [16, 32, 64]    
    YOLO_ANCHORS            = [[[10,  14], [23,   27], [37,   58]],
                               [[81,  82], [135, 169], [344, 319]],
                               [[0,    0], [0,     0], [0,     0]]]

def __read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

CLASS_NAMES=__read_class_names(YOLO_COCO_CLASSES)
NUM_CLASS = len(CLASS_NAMES)
params = locals()