"""
Helper functions
"""
import os
thread = "1"
os.environ["MKL_NUM_THREADS"] = thread
os.environ["NUMEXPR_NUM_THREADS"] = thread
os.environ["OMP_NUM_THREADS"] = thread
os.environ["VECLIB_MAXIMUM_THREADS"] = thread
os.environ["OPENBLAS_NUM_THREADS"] = thread
import cv2 as cv
cv.setNumThreads(0)

import json
from collections import defaultdict
from random import sample
import torch
import numpy as np
from moviepy.editor import ImageSequenceClip, clips_array
from os import listdir
from os.path import isfile, join, isdir
import requests


# Check if a file exists
def is_file_here(file_path):
    return os.path.isfile(file_path)


# Check if a directory exists, if not, create it
def check_and_create_dir(path):
    if path is None: return
    dir_name = os.path.dirname(path)
    if dir_name != "" and not os.path.exists(dir_name):
        try: # this is used to prevent race conditions during parallel computing
            os.makedirs(dir_name)
        except Exception as ex:
            print(ex)


# Return a list of all files in a folder
def get_all_file_names_in_folder(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


# Return a list of all directories in a folder
def get_all_dir_names_in_folder(path):
    return [f for f in listdir(path) if isdir(join(path, f))]


# Load json file
def load_json(fpath):
    with open(fpath, "r") as f:
        return json.load(f)


# Save json file
def save_json(content, fpath):
    with open(fpath, "w") as f:
        json.dump(content, f)


# Request json from url
def request_json(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        return None


# Convert a defaultdict to dict
def ddict_to_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict_to_dict(v)
    return dict(d)


# Compute a confusion matrix of samples
# The first key is the true label
# The second key is the predicted label
# Input:
#   y_true (list): true labels
#   y_pred (list): predicted labels
#   n (int):
#       minimum number of samples to return for each cell in the matrix
#       if n=None, will return all samples
# Output:
#   (dictionary):
#       the first key is the true label
#       the second key is the predicted label
def confusion_matrix_of_samples(y_true, y_pred, n=None):
    if len(y_true) != len(y_pred):
        print("Error! y_true and y_pred have different lengths.")
        return
    if y_true is None or y_pred is None:
        print("Error! y_true or y_pred is None.")
        return

    # Build the confusion matrix
    cm = defaultdict(lambda: defaultdict(list))
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]].append(i)

    # Randomly sample the confusion matrix
    if n is not None:
        for u in cm:
            for v in cm[u]:
                s = cm[u][v] # get the items
                if len(s) > n: # need to sample from the items
                    cm[u][v] = sample(s, n)

    return ddict_to_dict(cm)


# Write video data summary to files
# Input:
#   cm (dict): the confusion matrix returned by the confusion_matrix_of_samples function
#   file_name (list): a list of file names for the rgb or optical flow frames
#   p_frame (str): path to the rgb or optical flow frames
#   p_save (str): path to save the video
#   global_step (int): the training step of the model
def write_video_summary(cm, file_name, p_frame, p_save, global_step=None, fps=12):
    check_and_create_dir(p_save)
    for u in cm:
        for v in cm[u]:
            tag = "true_%d_prediction_%d" % (u, v)
            if global_step is not None:
                tag += "_step_%d" % global_step
            grid_x = []
            grid_y = []
            items = cm[u][v]
            for idx in items:
                frames = np.load(p_frame + file_name[idx] + ".npy")
                shape = frames.shape
                if shape[3] == 2: # this means that the file contains optical flow frames (x and y)
                    tmp = np.zeros((shape[0], shape[1], shape[2], 3), dtype=np.float64)
                    for i in range(shape[0]):
                        # To visualize the flow, we need to first convert flow x and y to hsv
                        flow_x = frames[i, :, :, 0]
                        flow_y = frames[i, :, :, 1]
                        magnitude, angle = cv.cartToPolar(flow_x / 255, flow_y / 255, angleInDegrees=True)
                        tmp[i, :, :, 0] = angle # channel 0 represents direction
                        tmp[i, :, :, 1] = 1 # channel 1 represents saturation
                        tmp[i, :, :, 2] = magnitude # channel 2 represents magnitude
                        # Convert the hsv to rgb
                        tmp[i, :, :, :] = cv.cvtColor(tmp[i, :, :, :].astype(np.float32), cv.COLOR_HSV2RGB)
                    frames = tmp
                else: # this means that the file contains rgb frames
                    frames = frames / 255 # tensorboard needs the range between 0 and 1
                if frames.dtype != np.uint8:
                    frames = (frames * 255).astype(np.uint8)
                frames = ImageSequenceClip([I for I in frames], fps=12)
                grid_x.append(frames)
                if len(grid_x) == 8:
                    grid_y.append(grid_x)
                    grid_x = []
            if len(grid_x) != 0:
                grid_y.append(grid_x)
            if len(grid_y) > 1 and len(grid_y[-1]) != len(grid_y[-2]):
                grid_y = grid_y[:-1]
            try:
                clips_array(grid_y).write_videofile(p_save + tag + ".mp4")
            except Exception as ex:
                for a in grid_y:
                    print(len(a))
                print(ex)
