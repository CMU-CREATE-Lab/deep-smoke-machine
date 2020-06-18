import os
import sys
from util import *
import numpy as np
from optical_flow.optical_flow import OpticalFlow
import urllib.request
from i3d_learner import I3dLearner
from grad_cam_viz import GradCam
import torch.nn.functional as F
import torch
import re
from datetime import datetime
from urllib.parse import urlparse, parse_qs


# This is the production code for smoke recognition
def main(argv):
    vid_p = "../data/production/video/"
    rgb_p = "../data/production/rgb/"
    check_and_create_dir(rgb_p)
    check_and_create_dir(vid_p)
    url = "https://thumbnails-v2.createlab.org/thumbnail?root=http://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2019-02-03.timemachine/&boundsLTRB=5329,953,5831,1455&width=180&height=180&startFrame=7748&format=mp4&fps=12&tileFormat=mp4&nframes=360"
    url_part_list, file_name_list = gen_url_parts(url)
    if url_part_list is None or file_name_list is None:
        print("Error generating url parts...END")
    url_root = "https://thumbnails-v2.createlab.org/thumbnail"
    for i in range(len(url_part_list)):
        fn = file_name_list[i]
        url = url_root + url_part_list[i]
        if is_file_here(rgb_p + fn + ".npy"):
            print("File exists...skip downloading...")
        else:
            if download_video(url, vid_p, fn):
                video_to_numpy(url, vid_p, rgb_p, fn)
            else:
                print("Error downloading video...END")
                return
    recognize_smoke(rgb_p, file_name_list)
    print("END")


# Parse the datetime string from the thumbnail server url
def get_datetime_str_from_url(url):
    m = re.search("\d+-\d+-\d+\.timemachine", url)
    return m.group(0).split(".")[0]


# Parse the camera name from the thumbnail server url
def get_camera_name_from_url(url):
    m = re.search("tiles.cmucreatelab.org/ecam/timemachines/\w+/", url)
    return m.group(0).split("/")[3]


# Parse the bounding box from the thumbnail server url
def get_bound_from_url(url):
    b_str = parse_qs(urlparse(url).query)["boundsLTRB"][0]
    b_str_split = list(map(int, b_str.split(",")))
    return {"L": b_str_split[0], "T": b_str_split[1], "R": b_str_split[2], "B": b_str_split[3]}


# Get the url that contains the information about the frame captured times
def get_tm_json_url(cam_name=None, ds=None):
    return "https://tiles.cmucreatelab.org/ecam/timemachines/%s/%s.timemachine/tm.json" % (cam_name, ds)


# Convert string to datetime object
def str_to_time(ds):
    return datetime.strptime(ds, "%Y-%m-%d %H:%M:%S")


# Given a frame captured time array (from time machine)
# Divide it into a set of starting frames
# - nf: number of frames of each divided video
def divide_start_frame(ct_list, nf=360):
    # The sun set and rise time in Pittsburgh
    # Format: [[Jan_sunrise, Jan_sunset], [Feb_sunrise, Feb_sunset], ...]
    pittsburgh_sun_table = [(8,16), (8,17), (8,18), (7,19), (6,19), (6,20), (6,19), (7,19), (7,18), (8,17), (8,16), (8,16)]

    # Compute the list of starting frames
    # Each starting frame is for one divided video
    sunset = None
    sunrise = None
    frame_min = None
    frame_max = None
    for i in range(len(ct_list)):
        dt = str_to_time(ct_list[i])
        if sunset is None:
            sunrise, sunset = pittsburgh_sun_table[dt.month - 1]
        if frame_min is None and dt.hour >= sunrise:
            frame_min = i + 1
        if frame_max is None and dt.hour == sunset + 1:
            frame_max = i
            break
    if frame_min is None:
        return (None, None, None)
    if frame_max is None:
        frame_max = len(ct_list)
    r = range(frame_min, frame_max, nf)
    if len(r) == 0:
        return (None, None, None)

    # Compute the starting and ending datetime list
    sf_list = []
    sf_dt_list = []
    ef_dt_list = []
    for sf in r:
        ef = sf + nf - 1 # end frame
        if ef > frame_max: break
        sf_list.append(sf)
        sf_dt_list.append(str_to_time(ct_list[sf]))
        ef_dt_list.append(str_to_time(ct_list[ef]))
    return (sf_list, sf_dt_list, ef_dt_list)


# Return a thumbnail server url part
# - cam_name: camera name (str), e.g., "clairton1"
# - ds: datetime string (str), "2015-05-22"
# - b: bounding box (dictionary with Left Top Right Bottom), e.g., {"L": 2330, "T": 690, "R": 3730, "B": 2090}
# - w: width (int)
# - h: height (int)
# - sf: starting frame number (int)
# - fmt: format (str), "gif" or "mp4" or "png"
# - fps: frames per second (int)
# - nf: number of frames (int)
def get_url_part(cam_name=None, ds=None, b=None, w=180, h=180, sf=None, fmt="mp4", fps=12, nf=360):
    return "?root=http://tiles.cmucreatelab.org/ecam/timemachines/%s/%s.timemachine/&boundsLTRB=%r,%r,%r,%r&width=%r&height=%r&startFrame=%r&format=%s&fps=%r&tileFormat=mp4&nframes=%r" % (cam_name, ds, b["L"], b["T"], b["R"], b["B"], w, h, sf, fmt, fps, nf)


# Return a file name
# - cam_id: camera id (int)
# - ds: datetime string (str), "2015-05-22"
# - b: bounding box (dictionary with Left Top Right Bottom), e.g., {"L": 2330, "T": 690, "R": 3730, "B": 2090}
# - w: width (int)
# - h: height (int)
# - sf: starting frame number (int)
# - st: starting epochtime in seconds (int)
# - et: ending epochtime in seconds (int)
def get_file_name(cam_id, ds, b, w, h, sf, st, et):
    return "%d-%s-%r-%r-%r-%r-%r-%r-%r-%r-%r" % (cam_id, ds, b["L"], b["T"], b["R"], b["B"], w, h, sf, st, et)


# Convert the camera name to camera id
def cam_name_to_id(name):
    if name == "clairton1":
        return 0
    elif name == "braddock1":
        return 1
    elif name == "westmifflin1":
        return 2
    else:
        return None


# Given a thumbnail url (having the date, camera, and bounding box information)
# Generate the followings:
# - a list of thumbnail url parts for that date, camera, and bounding box
# - a list of file names, corresponding to the url parts
def gen_url_parts(url, video_size=180):
    # Get frame captured times
    ds = get_datetime_str_from_url(url)
    cam_name = get_camera_name_from_url(url)
    tm_json = request_json(get_tm_json_url(cam_name=cam_name, ds=ds))
    if tm_json is None:
        print("Error getting frame captured times")
        return None

    # Divide the large video into small ones
    sf_list, sf_dt_list, ef_dt_list = divide_start_frame(tm_json["capture-times"], nf=360)
    if sf_list is None:
        print("Error dividing videos")
        return (None, None)

    # Generate the url parts and file names
    cam_id = cam_name_to_id(cam_name)
    if cam_id is None:
        print("Error finding camera id for camera name:", cam_name)
        return (None, None)
    b = get_bound_from_url(url)
    url_part_list = []
    file_name_list = []
    for i in range(len(sf_list)):
        sf = sf_list[i]
        url_part_list.append(get_url_part(cam_name=cam_name, ds=ds, b=b, sf=sf, w=video_size, h=video_size))
        st = int(sf_dt_list[i].timestamp())
        et = int(ef_dt_list[i].timestamp())
        file_name_list.append(get_file_name(cam_id, ds, b, video_size, video_size, sf, st, et))
    return (url_part_list, file_name_list)


# Given a video url (the thumbnail server)
# Download and save the video to a local file
def download_video(url, vid_p, file_name):
    try:
        print("Downloading video:", url)
        urllib.request.urlretrieve(url, vid_p + file_name + ".mp4")
    except:
        print("Error downloading:", url)
        return False
    return True


# Load the video and convert it to numpy.array
# Then save the numpy.array to a local file
def video_to_numpy(url, vid_p, rgb_p, file_name):
    rgb_vid_in_p = vid_p + file_name + ".mp4"
    rgb_4d_out_p = rgb_p + file_name + ".npy"
    op = OpticalFlow(rgb_vid_in_p=rgb_vid_in_p, rgb_4d_out_p=rgb_4d_out_p, flow_type=None)
    op.process()


# The core function for smoke recognition
# Input:
# - rgb_p: file path that points to the numpy.array file
# - file_name_list: list of file names of the numpy.array file
# - the video numpy.array has shape (time, height, width, channel) = (n, 180, 180, 3)
# First output:
# - probabilities of having smoke, with shape (time, num_of_classes) = (n, 2)
# Second output:
# - the number of smoke pixels (using Grad-CAM), with shape (time, num_of_smoke_pixels)
def recognize_smoke(rgb_p, file_name_list):
    # Prepare model
    mode = "rgb"
    use_cuda = True
    if use_cuda:
        print("Enable GPU computing...")
    learner = I3dLearner(mode=mode, use_cuda=use_cuda, parallel=False)
    p_model = "../data/saved_i3d/paper_result/full-augm-rgb/55563e4-i3d-rgb-s3/model/573.pt"
    model = learner.set_model(0, 1, mode, p_model, False)
    transform = learner.get_transform(mode, image_size=224)

    # Iterate
    for fn in file_name_list:
        print("Process file:", fn)
        # Compute probability
        print("Compute probability...")
        v = np.load(rgb_p + fn + ".npy")
        v = transform(v)
        v = torch.unsqueeze(v, 0)
        if use_cuda and torch.cuda.is_available:
            v = v.cuda()
        pred = learner.make_pred(model, v).squeeze().transpose(0, 1)
        pred = F.softmax(pred).cpu().detach().numpy()
        print(pred.shape)
        # Grad Cam
        print("Compute number of smoke pixels...")
        grad_cam = GradCam(model, use_cuda=use_cuda)
        target_class = 1 # has smoke
        cam = grad_cam.generate_cam(v, target_class)
        print(cam.shape)


if __name__ == "__main__":
    main(sys.argv)
