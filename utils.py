# -----------------------------------------------------------------------------
# Utility functions to load modules
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------
# Script to define functions called to load the modules

import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn import preprocessing
import sys

sys.path.append('./band')
from band6 import *
from const import *

def load_model():

    model = tf.keras.models.load_model("bestmodelalexnet.keras")

    return model

def load_detector():

    base_options = python.BaseOptions(model_asset_path='detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    return detector

def load_band():

    mac_add = "D1:83:95:98:39:F1"

    with open("band/auth_key.txt") as f:
        auth_key = f.read()

    manager = gatt.DeviceManager(adapter_name='hci0')
    device = MiBand6(mac_address = mac_add, manager = manager)

    return device, manager, auth_key

