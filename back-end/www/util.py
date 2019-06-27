"""
Helper functions
"""

import json
import os
from collections import defaultdict
from random import sample
import torch
import numpy as np
import cv2 as cv

# Check if a file exists
def is_file_here(file_path):
    return os.path.isfile(file_path)

# Check if a directory exists, if not, create it
def check_and_create_dir(path):
    if path is None: return
    dir_name = os.path.dirname(path)
    if dir_name != "" and not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Load json file
def load_json(fpath):
    with open(fpath, "r") as f:
        return json.load(f)

# Save json file
def save_json(content, fpath):
    with open(fpath, "w") as f:
        json.dump(content, f)

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
#   n_min (int): minimum number of samples to return for each cell in the matrix
def confusion_matrix_of_samples(y_true, y_pred, n=32):
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

# Write video data summary to the tensorboard
#   writer (torch.utils.tensorboard.SummaryWriter): tensorboard summary writer
#   cm (dict): the confusion matrix returned by the confusion_matrix_of_samples function
#   file_name (list): a list of file names for the rgb or optical flow frames
#   p_frame (str): path to the rgb or optical flow frames
#   global_step (int): the current training step
def write_video_summary(writer, cm, file_name, p_frame, global_step=None, fps=12):
    for u in cm:
        for v in cm[u]:
            tag = "true_%d_prediction_%d" % (u, v)
            if global_step is not None:
                tag += "_step_%d" % global_step
            grid = []
            items = cm[u][v]
            for idx in items:
                frames = np.load(p_frame + file_name[idx] + ".npy")
                shape = frames.shape
                if shape[3] == 2: # this means that the file is optical flows (x and y)
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
                        tmp[i, :, :, :] = cv.cvtColor(tmp[i, :, :, :], cv.COLOR_HSV2RGB)
                    frames = tmp
                frames = frames / 255 # tensorboard needs the range between 0 and 1
                frames = frames.transpose([0,3,1,2])
                grid.append(frames)
            grid = torch.from_numpy(np.array(grid))
            writer.add_video(tag, grid, fps=fps)
