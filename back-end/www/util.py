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
import uuid
import copy


# Given an array of zeros and ones, output a list of events [[start_1, end_1], [start_2, end_2], ...]
# (start_i and end_i means the starting and ending index in the array for each event)
# An event means continuous ones in the array
# For example, if array=[0,0,0,1,1,1,1,0,0,0,0,1,1,1], event=[[3,6],[11,13]]
# Input:
# - array: an array of zeros and ones (e.g., [0,0,0,1,1,1,1,0,0,0,0,1,1,1])
# - max_len: the max length of the event (e.g., when max_len=2 we get event=[[3,4],[5,6],[11,12],[13,13]])
# TODO: need to implement an option called stride
# TODO: by default stride=0
# TODO: if stride=1 and max_len=2, then we get event=[[3,4],[6,6],[11,12]], where 5 and 13 are ignored due to the stride
def array_to_event(array, max_len=None):
    event = []
    array = copy.deepcopy(array)
    array.insert(0, 0) # insert a zero at the begining
    if max_len is not None and max_len < 1: max_len = None
    for i in range(len(array)-1):
        a_i1 = array[i+1]
        diff = a_i1 - array[i]
        if diff == 1: # from 0 to 1
            event.append([i,i])
            if max_len == 1:
                array[i+1] = 0 # restart next event
        elif diff == 0: # from 0 to 0, or from 1 to 1
            if a_i1 == 1: # from 1 to 1
                event[-1][1] = i
                if max_len is not None and i-event[-1][0]+1 >= max_len:
                    array[i+1] = 0 # restart next event
    return event


# Test the array_to_event function
def test_array_to_event():
    test_cases = [
            ([], None, []),
            ([0], None, []),
            ([1], None, [[0,0]]),
            ([0,0,1,1], None, [[2,3]]),
            ([1,1,1,0,0], None, [[0,2]]),
            ([0,0,1,1,1,0,0,0], None, [[2,4]]),
            ([1,1,1,0,1,1], None, [[0,2],[4,5]]),
            ([0,0,1,1,1,0,0,0,1,1,0,1], None, [[2,4],[8,9],[11,11]]),
            ([], 3, []),
            ([0], 3, []),
            ([1], 3, [[0,0]]),
            ([0,0,1,1], 3, [[2,3]]),
            ([0,0,1,1], 1, [[2,2],[3,3]]),
            ([1,1,1,0,0], 3, [[0,2]]),
            ([1,1,1,1,0,0], 4, [[0,3]]),
            ([1,1,1,1,0,0], 3, [[0,2],[3,3]]),
            ([1,1,1,1,0,0], 2, [[0,1],[2,3]]),
            ([1,1,1,1,0,0], 1, [[0,0],[1,1],[2,2],[3,3]]),
            ([0,0,1,1,1,0,0,0], 4, [[2,4]]),
            ([0,0,1,1,1,0,0,0], 3, [[2,4]]),
            ([0,0,1,1,1,0,0,0], 2, [[2,3],[4,4]]),
            ([0,0,1,1,1,0,0,0], 1, [[2,2],[3,3],[4,4]]),
            ([0,1,1,1,1,1,0,0,0], 3, [[1,3],[4,5]]),
            ([1,1,1,0,1,1], 4, [[0,2],[4,5]]),
            ([1,1,1,0,1,1], 3, [[0,2],[4,5]]),
            ([1,1,1,0,1,1], 2, [[0,1],[2,2],[4,5]]),
            ([1,1,1,0,1,1], 1, [[0,0],[1,1],[2,2],[4,4],[5,5]]),
            ([0,0,1,1,1,0,0,0,1,1,0,1], 3, [[2,4],[8,9],[11,11]]),
            ([0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,1], 3, [[2,4],[8,10],[11,13],[14,15],[17,17]]),
            ([1,1,1,1,1,1,1,1,1,1], 3, [[0,2],[3,5],[6,8],[9,9]]),
    ]
    for c in test_cases:
        output_c = array_to_event(c[0], max_len=c[1])
        if output_c == c[2]:
            print("pass")
        else:
            print("WRONG!")
            print("Input: %r max_len=%r" % (c[0], c[1]))
            print("Output: %r" % array_to_event(c[0], max_len=c[1]))
            print("Desired output: %r" % c[2])
            print("-"*50)


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


# Return the root url for ESDR
def esdr_root_url():
    return "https://esdr.cmucreatelab.org/"


# Get the access token from ESDR
# For details, see https://github.com/CMU-CREATE-Lab/esdr/blob/master/HOW_TO.md
# Input:
#   auth_json: a dictionary for authentication
def get_esdr_access_token(auth_json):
    url = esdr_root_url() + "oauth/token"
    headers = {"Authorization": "", "Content-Type": "application/json"}
    r = requests.post(url, data=json.dumps(auth_json), headers=headers)
    r_json = r.json()
    if r.status_code is not 200:
        print("ERROR! ESDR returns:" + json.dumps(r_json))
        return None, None
    else:
        access_token = r_json["access_token"]
        user_id = r_json["userId"]
        print("Receive access token " + access_token)
        print("Receive user ID " + str(user_id))
        return access_token, user_id


# Register a product on ESDR
# For details, see https://github.com/CMU-CREATE-Lab/esdr/blob/master/HOW_TO.md
# Input:
#   product_json: a dictionary that specifies the product (i.e., the data format)
#   access_token: the access token, obtained by calling the get_esdr_access_token() function
def register_esdr_product(product_json, access_token):
    headers = {"Authorization": "Bearer " + access_token, "Content-Type": "application/json"}
    url = esdr_root_url() + "api/v1/products"
    r = requests.post(url, data=json.dumps(product_json), headers=headers)
    print("ESDR returns: %r" % r.content)


# Upload data to ESDR
# data_json = {
#   "channel_names": ["particle_concentration", "particle_count", "raw_particles", "temperature"],
#   "data": [[1449776044, 0.3, 8.0, 6.0, 2.3], [1449776104, 0.1, 3.0, 0.0, 4.9]]
# }
def upload_data_to_esdr(device_name, data_json, product_id, access_token, **options):
    # Set the header for http request
    headers = {"Authorization": "Bearer " + access_token, "Content-Type": "application/json"}

    # Check if the device exists
    print("\tTry getting the device ID of device name '" + device_name + "'")
    url = esdr_root_url() + "api/v1/devices?where=name=" + device_name + ",productId=" + str(product_id)
    r = requests.get(url, headers=headers)
    r_json = r.json()
    device_id = None
    print("\tESDR returns: " + json.dumps(r_json) + " when getting the device ID for '" + device_name + "'")
    if r.status_code is 200:
        if r_json["data"]["totalCount"] < 1:
            print("\t'" + device_name + "' did not exist")
        else:
            device_id = r_json["data"]["rows"][0]["id"]
            print("\tReceive existing device ID " + str(device_id))

    # Create a device if it does not exist
    if device_id is None:
        print("\tCreate a device for '" + device_name + "'")
        url = esdr_root_url() + "api/v1/products/" + str(product_id) + "/devices"
        device_json = {
            "name": device_name,
            "serialNumber": options["serialNumber"] if "serialNumber" in options else str(uuid.uuid4())
        }
        r = requests.post(url, data=json.dumps(device_json), headers=headers)
        r_json = r.json()
        print("\tESDR returns: " + json.dumps(r_json) + " when creating a device for '" + device_name + "'")
        if r.status_code is 201:
            device_id = r_json["data"]["id"]
            print("\tCreate new device ID " + str(device_id))
        else:
            return None

    # Check if a feed exists for the device
    print("\tGet feed ID for '" + device_name + "'")
    url = esdr_root_url() + "api/v1/feeds?where=deviceId=" + str(device_id)
    r = requests.get(url, headers=headers)
    r_json = r.json()
    feed_id = None
    api_key = None
    api_key_read_only = None
    print("\tESDR returns: " + json.dumps(r_json) + " when getting the feed ID")
    if r.status_code is 200:
        if r_json["data"]["totalCount"] < 1:
            print("\tNo feed ID exists for device " + str(device_id))
        else:
            row = r_json["data"]["rows"][0]
            feed_id = row["id"]
            api_key = row["apiKey"]
            api_key_read_only = row["apiKeyReadOnly"]
            print("\tReceive existing feed ID " + str(feed_id))

    # Create a feed if no feed ID exists
    if feed_id is None:
        print("\tCreate a feed for '" + device_name + "'")
        url = esdr_root_url() + "api/v1/devices/" + str(device_id) + "/feeds"
        feed_json = {
            "name": device_name,
            "exposure": options["exposure"] if "exposure" in options else "virtual",
            "isPublic": options["isPublic"] if "isPublic" in options else 0,
            "isMobile": options["isMobile"] if "isMobile" in options else 0,
            "latitude": options["latitude"] if "latitude" in options else None,
            "longitude": options["longitude"] if "longitude" in options else None
        }
        r = requests.post(url, data=json.dumps(feed_json), headers=headers)
        r_json = r.json()
        print("\tESDR returns: " + json.dumps(r_json) + " when creating a feed")
        if r.status_code is 201:
            feed_id = r_json["data"]["id"]
            api_key = r_json["data"]["apiKey"]
            api_key_read_only = r_json["data"]["apiKeyReadOnly"]
            print("\tCreate new feed ID " + str(feed_id))
        else:
            return None

    # Upload Speck data to ESDR
    print("\tUpload sensor data for '" + device_name + "'")
    url = esdr_root_url() + "api/v1/feeds/" + str(feed_id)
    r = requests.put(url, data=json.dumps(data_json), headers=headers)
    r_json = r.json()
    print("\tESDR returns: " + json.dumps(r_json) + " when uploading data")
    if r.status_code is not 200:
        return None

    # Return a list of information for getting data from ESDR
    print("\tData uploaded")
    return (device_id, feed_id, api_key, api_key_read_only)


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
