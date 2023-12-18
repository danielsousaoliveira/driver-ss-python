# -----------------------------------------------------------------------------
# Utility functions for preprocessing dataset
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------
# Script to define utility functions

import os
import cv2
import random
import shutil
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def category_to_id(category):

    categories = ["normal", "drowsy", "distracted"]
    id = -1

    for i, c in enumerate(categories):
        if c == category:
            id = i

    return id

def situation_to_id(situation):

    situationS = ["glasses", "noglasses", "nightglasses", "nightnoglasses", "sunglasses"]
    id = -1
    
    for i, s in enumerate(situationS):
        if s == situation:
            id = i

    return id

def mediapipe_face_detector(source_folder, destination_folder):

    base_options = python.BaseOptions(model_asset_path='detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    for category_folder in os.listdir(source_folder):
        category_path = os.path.join(source_folder, category_folder)

        if os.path.isdir(category_path):
            print(f"Processing category: {category_folder}")

            for frame in os.listdir(category_path):
                detected_face = None
                image_path = os.path.join(category_path, frame)
                image_array = cv2.imread(image_path)

                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)

                detection_result = detector.detect(image)

                for detection in detection_result.detections:

                    bbox = detection.bounding_box
                    detected_face = image_array[int(bbox.origin_y):int(bbox.origin_y + bbox.height), int(bbox.origin_x):int(bbox.origin_x + bbox.width)].copy()
                
                output_path = os.path.join(destination_folder, category_folder)
                output_path = os.path.join(output_path, frame)

                if detected_face is not None and detected_face.size>0:
                    cv2.imwrite(output_path, detected_face)

    print("Frame selection and copying completed.")

def over_sampling(root_dir):

    newnumber = 76215

    category = "distracted"
    category_dir = os.path.join(root_dir, category)
    
    files = os.listdir(category_dir)

    desired_samples = random.randint(7600,7700)

    oversample_files = random.sample(files, desired_samples)
    for oversample_file in oversample_files:
        source_path = os.path.join(category_dir, oversample_file)
        new_filename = oversample_file.split("_")[0] + "_" + oversample_file.split("_")[1] + "_frame_" + str(newnumber) + ".png"
        destination_path = os.path.join(category_dir, new_filename)
        shutil.copy(source_path, destination_path)
        newnumber += 1

def under_sampling(root_dir):

    categories = ["drowsy", "normal"]

    for category in categories:

        category_dir = os.path.join(root_dir, category)
        files = os.listdir(category_dir)
        
        if category == "drowsy":
            desired_samples = int(len(files)/2) + 1
        else:
            desired_samples = random.randint(35300,35400)

        files = os.listdir(category_dir)
        undersample_files = random.sample(files, desired_samples)
        for undersample_file in undersample_files:
            file_path = os.path.join(category_dir, undersample_file)
            os.remove(file_path)

def split_data(root_dir):

    dataset_directory = "/home/daniel/final"

    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'

    train_split = 0.7
    val_split = 0.15

    os.makedirs(os.path.join(dataset_directory, train_dir), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, val_dir), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, test_dir), exist_ok=True)

    for category_folder in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_folder)

        if os.path.isdir(category_path):
            image_files = os.listdir(category_path)
            random.shuffle(image_files)

            num_images = len(image_files)
            num_train = int(num_images * train_split)
            num_val = int(num_images * val_split)

            train_images = image_files[:num_train]
            val_images = image_files[num_train:num_train + num_val]
            test_images = image_files[num_train + num_val:]

            for image in train_images:
                src_path = os.path.join(category_path, image)
                dest_path = os.path.join(dataset_directory, train_dir, category_folder, image)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)

            for image in val_images:
                src_path = os.path.join(category_path, image)
                dest_path = os.path.join(dataset_directory, val_dir, category_folder, image)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)

            for image in test_images:
                src_path = os.path.join(category_path, image)
                dest_path = os.path.join(dataset_directory, test_dir, category_folder, image)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)

    print("Dataset split completed.")

def label_and_preprocess_frames(root_dir, dest_dir):

    frames_count = people_count = 0
    category_counter = np.zeros(3, int)
    situation_counter = np.zeros(5, int)


    for person_folder in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person_folder)
        if not os.path.isdir(person_path):
            continue
        
        personID = person_folder[:3]
        people_count += 1

        for situation_folder in os.listdir(person_path):
            situation_path = os.path.join(person_path, situation_folder)
            if not os.path.isdir(situation_path):
                continue

            for category_folder in os.listdir(situation_path):
                category_path = os.path.join(situation_path, category_folder)
                if not os.path.isdir(category_path):
                    continue

                situationID = situation_to_id(situation_folder)
                
                for frame_filename in os.listdir(category_path):

                    frame_path = os.path.join(category_path, frame_filename)

                    frame = cv2.imread(frame_path)                   

                    width, height = 224, 224
                    frame = cv2.resize(frame, (width, height))

                    output_file_name = f"{personID}_{situationID}_frame_{frames_count}.png"

                    newcategory = category_folder

                    output_path = os.path.join(dest_dir, newcategory)
                    output_path = os.path.join(output_path, output_file_name)

                    cv2.imwrite(output_path, frame)

                    situationID = situation_to_id(situation_folder)
                    categoryID = category_to_id(newcategory)

                    situation_counter[situationID] += 1
                    category_counter[categoryID] += 1

                    frames_count += 1
    
    print("Total number of people: " + str(people_count))
    print("Total number of frames: " + str(frames_count) + "\n")
    print("Number of situations: " + str(len(situation_counter)) )
    print("Situation No Glasses: " + str(situation_counter[1]) + " frames")
    print("Situation With Glasses: " + str(situation_counter[0]) + " frames")
    print("Situation Night No Glasses: " + str(situation_counter[3]) + " frames")
    print("Situation Night With Glasses: " + str(situation_counter[2]) + " frames")
    print("Situation Sunglasses: " + str(situation_counter[4]) + " frames \n")
    print("Number of categories: " + str(len(category_counter)))
    print("Category Normal: " + str(category_counter[0]) + " frames")
    print("Category Drowsy: " + str(category_counter[1]) + " frames")
    print("Category Distracted: " + str(category_counter[2]) + " frames")

def print_dataset(root_dir):

    category_counter = np.zeros(3, int)

    for category_folder in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_folder)
        if not os.path.isdir(category_path):
            continue
        
        for frame_filename in os.listdir(category_path):

            categoryID = category_to_id(category_folder)
            category_counter[categoryID] += 1

    print("Total number of frames: " + str(sum(category_counter)) + "\n")
    print("Category Normal: " + str(category_counter[0]) + " frames")
    print("Category Drowsy: " + str(category_counter[1]) + " frames")
    print("Category Distracted: " + str(category_counter[2]) + " frames \n")

def show_random_sample(root_dir):

    selected_frames = []
    selected_categories = []

    for category_folder in os.listdir(root_dir):

        category_path = os.path.join(root_dir, category_folder)
        if not os.path.isdir(category_path):
            continue
        
        frames = os.listdir(category_path)
        selected_frame = random.choice(frames)

        frame_path = os.path.join(category_path, selected_frame)

        frame = cv2.imread(frame_path)
        
        selected_frames.append(frame)
        
        if category_folder == "normal":
            selected_categories.append("Normal")
        elif category_folder == "distracted":
            selected_categories.append("Distracted")
        else:
            selected_categories.append("Drowsy")

    _, axs = plt.subplots(1, 3)

    for j in range(3):
        axs[j].imshow(selected_frames[j])
        axs[j].set_title(selected_categories[j])
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()