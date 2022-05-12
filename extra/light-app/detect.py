#================================================================
# Yolo V-3 by
# Email         : irfanhasib.me@gmail.com
# GitHub      : https://github.com/irfanhasib0/Deep-Learning-For-Robotics
#================================================================
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(0)
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import time
import glob
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue, Pipe
import time
import shutil
import json
from tqdm import tqdm,trange
import pickle
import zlib
from datetime import datetime

from yolo.model import YoloModel, calc_yolo_loss, calc_seg_loss
from yolo.decoder import YoloDecodeNetout
from yolo.dataset import Dataset
from yolo.eval import get_mAP
from yolo.utils import Utils
from yolo.seg_loader import Seg_Utils
from yolo.config import *
from yolo.tf import *

import socket
import cv2
import pickle
import struct
from collections import deque

yolo = YoloModel()
yolo_model=yolo.get_model()
decoder = YoloDecodeNetout()

def predict(frame):
    pred = decoder.detect_image(yolo_model, frame, output_path=TRAIN_CHECKPOINTS_FOLDER+'/pred_imgs/',input_size=256, show=True, score_threshold=0.3, iou_threshold=0.5, rectangle_colors='')
    return pred
