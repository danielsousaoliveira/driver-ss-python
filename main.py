# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------
# Load all modules and start the system

import cv2
import time
import threading
import queue
import signal
import sys
import numpy as np
import mediapipe as mp
from sklearn import preprocessing
from dbus.exceptions import DBusException
from utils import *


frame_queue = alert_queue = None
cap = total_processing_time = processed_frames = None
processing_thread = alert_thread = band_thread = None
processing = exit_flag = alert_flag = None

model = load_model()
detector = load_detector()
device, manager, auth_key = load_band()
min_max_scaler = preprocessing.MinMaxScaler()

def ping_band(device):

    global alert_flag

    if alert_flag:
        device.send_alert()

    device.ping_hr()
    return True


def preprocess_frame(frame):

    global detector, min_max_scaler

    detected_face = None
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    detection_result = detector.detect(image)

    for detection in detection_result.detections:

        bbox = detection.bounding_box
        detected_face = frame[int(bbox.origin_y):int(bbox.origin_y + bbox.height), int(bbox.origin_x):int(bbox.origin_x + bbox.width)].copy()

    if detected_face is None:
        return detected_face
    
    gray_frame = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    input_frame = cv2.resize(detected_face, (192, 192))
    input_frame = np.array(input_frame)
    input_frame = min_max_scaler.fit_transform(input_frame.reshape(-1, input_frame.shape[-1])).reshape(input_frame.shape)
    input_frame = np.expand_dims(input_frame, axis=0)
    
    return input_frame

def process_frames():

    global frame_queue, alert_queue, processing, total_processing_time, processed_frames, exit_flag

    while not exit_flag:
        if not frame_queue.empty():
            
            frame = frame_queue.get()
            if frame is None:
                break

            start_time = time.time()
            input_frame = preprocess_frame(frame)

            if input_frame is None:
                continue

            processing = True
            prediction = get_model_prediction(input_frame)
            alert_queue.put(prediction)
            
            processing = False

            end_time = time.time()
            
            processing_time = end_time - start_time
            total_processing_time += processing_time
            processed_frames += 1
        
    processing = False

def alert_function():

    global alert_queue, exit_flag, device, alert_flag

    while not exit_flag:
        counter = 0
        percentage = 0

        if alert_queue.qsize() == 120:

            total_elements = alert_queue.qsize()
            while not alert_queue.empty():
                element = alert_queue.get()
                if element != 2:
                    counter += 1

            percentage = (counter / total_elements)
        
            if percentage > 0.2:
                    
                print("ALERT - USER DROWSY OR DISTRACTED")
                alert_flag = True
                ping_band(device)

            else:
                alert_flag = False


def band_funtion():

    global exit_flag, device, manager, auth_key

    while not exit_flag:
        try:
            
            device.connect(auth_key)
            print("Connected")
            device.enable_notifications_chunked()
            manager.notification_query(ping_band,device)
            manager.run()

        except DBusException as e:
            print(f"Failed to connect to device : {e}")
            print("Retrying...\n")
            time.sleep(2)

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")

            device.disconnect()
            manager.stop()
            sys.exit(0)

        except AttributeError as e:
            print("Bluetooth problems found, try to restart your bluetooth")



def get_model_prediction(frame):
   
    global model

    prediction = model.predict(frame,verbose = 0)
    class_index = np.argmax(prediction)

    return class_index


def signal_handler(sig, frame):

    global frame_queue, processing_thread, cap, alert_queue, alert_thread, exit_flag, device, manager, band_thread

    print("Ctrl+C pressed. Stopping the program...")

    exit_flag = True

    alert_queue.put(None)
    alert_thread.join()
    frame_queue.put(None)
    processing_thread.join()

    device.print_hr()
    device.disconnect()
    manager.stop()
    band_thread.join()

    average_processing_time = total_processing_time / processed_frames
    print(f"Average processing time per frame: {average_processing_time:.4f} seconds")
    print(f"Average frames per second: {1/average_processing_time:.1f} fps")


    cap.release()
    cv2.destroyAllWindows()

    try:
        sys.exit(0)
    except SystemExit:
        print("Program terminated")

def main():

    global frame_queue, processing_thread, alert_thread, alert_queue, cap, processing, total_processing_time
    global processed_frames, band_thread, exit_flag, device

    frame_queue = queue.Queue()
    alert_queue = queue.Queue()
    total_processing_time = 0.0
    processed_frames = 0

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    processing_thread = threading.Thread(target=process_frames)
    alert_thread = threading.Thread(target=alert_function)
    band_thread = threading.Thread(target=band_funtion)
    processing_thread.start()
    alert_thread.start()
    band_thread.start()

    signal.signal(signal.SIGINT, signal_handler)

    time.sleep(10)

    while not device.authenticated:
        time.sleep(1)
        print("Waiting for band authentication")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if not frame_queue.full() and not processing:

            frame_queue.put(frame)

    exit_flag = True

if __name__ == "__main__":
    main()
