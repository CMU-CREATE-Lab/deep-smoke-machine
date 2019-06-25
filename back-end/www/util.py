"""
Helper functions
"""

import json
import os
from collections import defaultdict
from random import sample

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
# y_true: true labels
# y_pred: predicted labels
# n_min: minimum number of samples to return for each cell in the matrix
def confusion_matrix_of_samples(y_true, y_pred, n=16):
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
