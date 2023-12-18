# -----------------------------------------------------------------------------
# Utility functions for preprocessing dataset
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------
# Script to define utility functions

import numpy as np
import os
import cv2

def category_to_id(category):

    categories = ["normal", "drowsy", "distracted"]
    id = -1

    for i, c in enumerate(categories):
        if c == category:
            id = i

    return id

def label_binarizer(y):
  n = []
  classes = ["normal", "drowsy", "distracted"]
  binary = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

  for label in y:

      for i, c in enumerate(classes):

          if label == c:
              n.append(binary[i])
  return n

def load_dataset(dir, size):

    x = []
    y = []

    for category_folder in os.listdir(dir):

        category_path = os.path.join(dir, category_folder)

        if not os.path.isdir(category_path):
            continue

        for frames_folder in os.listdir(category_path):

                frame = cv2.imread(os.path.join(category_path, frames_folder))
                frame = cv2.resize(frame, size)
                x.append(frame)
                y.append(category_folder)

    return x, y

def print_dataset(root_dir):

    category_counter = np.zeros(3, int)

    for category_folder in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_folder)
        if not os.path.isdir(category_path):
            continue
        
        for _ in os.listdir(category_path):

            categoryID = category_to_id(category_folder)
            category_counter[categoryID] += 1

    print("Total number of frames: " + str(sum(category_counter)) + "\n")
    print("Category Normal: " + str(category_counter[0]) + " frames")
    print("Category Drowsy: " + str(category_counter[1]) + " frames")
    print("Category Distracted: " + str(category_counter[2]) + " frames \n")