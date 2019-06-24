"""
Helper functions
"""

import json
import os

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

# Compute a confusion matrix of samples
# y_true: true labels
# y_pred: predicted labels
# n_min: minimum number of samples to return for each cell in the matrix
def confusion_matrix_of_samples(y_true, y_pred, n=None):
    pass
