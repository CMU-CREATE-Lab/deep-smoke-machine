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
import pandas as pd
from multiprocessing.dummy import Pool
import time


def main(argv):
    if len(argv) < 2:
        print("Usage:")
        print("python recognize_smoke.py process_all_urls")
        print("python recognize_smoke.py init_data_upload")
        print("python recognize_smoke.py upload_data")
        return

    if argv[1] == "process_all_urls":
        process_all_urls()
    elif argv[1] == "init_data_upload":
        init_data_upload()
    elif argv[1] == "upload_data":
        upload_data()
    else:
        print("Wrong usage. Run 'python recognize_smoke.py' for details.")
    print("END")


# Register the product on the ESDR system
def init_data_upload():
    # Specify the data format (the definition of "product" on ESDR)
    product_json = {
      "name": "RISE_smoke_recognition_v1",
      "prettyName": "Recognizing Industrial Smoke Emissions",
      "vendor": "CMU CREATE Lab",
      "description": "Recognizing Industrial Smoke Emissions",
      "defaultChannelSpecs": {
        "version": 1,
        "channels": {
          "max_smoke_probability": {
            "prettyName": "max of the probability of having smoke",
            "units": "probability",
            "range": {
              "min": 0,
              "max": 1
            }
          },
          "mean_smoke_probability": {
            "prettyName": "mean of the probability of having smoke",
            "units": "probability",
            "range": {
              "min": 0,
              "max": 1
            }
          },
          "std_smoke_probability": {
            "prettyName": "std of the probability of having smoke",
            "units": "probability",
            "range": {
              "min": 0,
              "max": 1
            }
          },
          "max_activation_ratio": {
            "prettyName": "max of the ratio of activation region",
            "units": "ratio",
            "range": {
              "min": 0,
              "max": 1
            }
          },
          "mean_activation_ratio": {
            "prettyName": "mean of the ratio of activation region",
            "units": "ratio",
            "range": {
              "min": 0,
              "max": 1
            }
          },
          "std_activation_ratio": {
            "prettyName": "std of the ratio of activation region",
            "units": "ratio",
            "range": {
              "min": 0,
              "max": 1
            }
          }}}}

    # Get the ESDR access token
    access_token, _ = get_esdr_access_token(load_json("../data/auth.json"))

    # Register the product on ESDR
    if access_token is not None:
        register_esdr_product(product_json, access_token)
    else:
        print("ERROR! No access token.")


# Upload smoke recognition results to ESDR system
def upload_data():
    # Set product ID, obtained from the esdr response when calling register_esdr_product()
    product_id = 94 # this ID is for production

    # Get the access token
    access_token, _ = get_esdr_access_token(load_json("../data/auth.json"))
    if access_token is None:
        print("ERROR! No access token.")
        return

    # Upload all data
    p = "../data/production/"
    for dn in get_all_dir_names_in_folder(p): # date string
        for vn in get_all_dir_names_in_folder(p + dn + "/"): # camera view ID
            for fn in get_all_file_names_in_folder(p + dn + "/" + vn + "/"): # json file
                if ".json" not in fn: continue
                data = load_json(p + dn + "/" + vn + "/" + fn)
                if "channel_names" not in data or "data" not in data: continue
                s = vn.split("-")
                lat, lng = get_cam_location_by_id(int(s[0]))
                name = "RISE_smoke_recognition_v1_camera_%s_view_%s" % (s[0], s[1])
                upload_data_to_esdr(name, data, product_id, access_token, isPublic=1, latitude=lat, longitude=lng)


# Process all thumbnail server urls
def process_all_urls():
    p = "../data/production_url_list/"
    for fn in get_all_file_names_in_folder(p):
        if ".json" not in fn: continue
        m = load_json(p + fn)
        for m in load_json(p + fn):
            if "url" not in m or "cam_id" not in m or "view_id" not in m: continue
            process_url(m["url"], m["cam_id"], m["view_id"])


# Process each url and predict the probability of having smoke for that date and view
# Input:
#   url: the thumbnail server url that we want to process
#   cam_id: camera ID
#   view_id: view ID
def process_url(url, cam_id, view_id):
    print("Process %s" % url)
    url_root = "https://thumbnails-v2.createlab.org/thumbnail"

    # Divide the video into several small parts
    url_part_list, file_name_list, ct_sub_list, unique_p, uuid = gen_url_parts(url, cam_id, view_id, nf=36, overlap=18)
    if url_part_list is None or file_name_list is None:
        print("Error generating url parts...")
        return
    url_root = "https://thumbnails-v2.createlab.org/thumbnail"

    # The directory for saving the files
    p_root = "../data/production/" + unique_p
    vid_p = p_root + "video/"
    rgb_p = p_root + "rgb/"
    check_and_create_dir(rgb_p)
    check_and_create_dir(vid_p)

    # Skip if the json file exists
    if is_file_here(p_root + uuid + ".json"):
        print("File %s exists...skip..." % (uuid + ".json"))
        return

    # Download the videos
    download_video(url_part_list, file_name_list, url_root, vid_p)

    # Extract video frames
    for i in range(len(url_part_list)):
        fn = file_name_list[i]
        url = url_root + url_part_list[i]
        if is_file_here(rgb_p + fn + ".npy"):
            print("Frame file exists...skip...")
        else:
            video_to_numpy(url, vid_p, rgb_p, fn)

    # Apply the smoke recognition model on the video frames
    smoke_pb_list, activation_ratio_list = recognize_smoke(rgb_p, file_name_list)

    # Put data together for uploading to the ESDR system
    # Notice that for the epochtime, we use the ending time of the video (NOT starting time)
    # The reason is because we want consistent timestamps when doing real-time predictions
    data_json = {
        "channel_names": [
            "max_smoke_probability",
            "mean_smoke_probability",
            "std_smoke_probability",
            "max_activation_ratio",
            "mean_activation_ratio",
            "std_activation_ratio"
            ],
        "data": []
    }
    for i in range(len(smoke_pb_list)):
        smoke_pb = smoke_pb_list[i]
        activation_ratio = activation_ratio_list[i]
        ct_sub = ct_sub_list[i]
        v0 = int(np.max(ct_sub))
        v1 = round(float(np.max(smoke_pb)), 3)
        v2 = round(float(np.mean(smoke_pb)), 3)
        v3 = round(float(np.std(smoke_pb)), 3)
        v4 = round(float(np.max(activation_ratio)), 3)
        v5 = round(float(np.mean(activation_ratio)), 3)
        v6 = round(float(np.std(activation_ratio)), 3)
        data_item = [v0, v1, v2, v3, v4, v5, v6]
        data_json["data"].append(data_item)
    save_json(data_json, p_root + uuid + ".json")

    print("DONE process_url")


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
# Input:
#   ct_list: the captured time list
#   nf: number of frames of each divided video
#   overlap: the number of overlapping frames for each split
# Output:
#   sf_list: the starting frame number
#   sf_dt_list: the starting datetime
#   ef_dt_list: the ending datetime
#   ct_sub_list: the captured time array for each frame (in epochtime format)
def divide_start_frame(ct_list, nf=36, overlap=0):
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
    r = range(frame_min, frame_max, nf - overlap)
    if len(r) == 0:
        return (None, None, None)

    # Compute the starting and ending datetime list
    sf_list = [] # the starting frame number
    sf_dt_list = [] # the starting datetime
    ef_dt_list = [] # the ending datetime
    ct_sub_list = [] # the captured time array for each frame (in epochtime format)
    for sf in r:
        ef = sf + nf - 1 # end frame
        if ef > frame_max: break
        sf_list.append(sf)
        sf_dt_list.append(str_to_time(ct_list[sf]))
        ef_dt_list.append(str_to_time(ct_list[ef]))
        ct_item = []
        for i in range(sf, sf + nf):
            ct_item.append(str_to_time(ct_list[i]).timestamp())
        ct_item = list(map(int, ct_item))
        ct_sub_list.append(ct_item)
    return (sf_list, sf_dt_list, ef_dt_list, ct_sub_list)


# Return a thumbnail server url part
# Input:
#   cam_name: camera name (str), e.g., "clairton1"
#   ds: datetime string (str), "2015-05-22"
#   b: bounding box (dictionary with Left Top Right Bottom), e.g., {"L": 2330, "T": 690, "R": 3730, "B": 2090}
#   w: width (int)
#   h: height (int)
#   sf: starting frame number (int)
#   fmt: format (str), "gif" or "mp4" or "png"
#   fps: frames per second (int)
#   nf: number of frames (int)
def get_url_part(cam_name=None, ds=None, b=None, w=180, h=180, sf=None, fmt="mp4", fps=12, nf=36):
    return "?root=http://tiles.cmucreatelab.org/ecam/timemachines/%s/%s.timemachine/&boundsLTRB=%r,%r,%r,%r&width=%r&height=%r&startFrame=%r&format=%s&fps=%r&tileFormat=mp4&nframes=%r" % (cam_name, ds, b["L"], b["T"], b["R"], b["B"], w, h, sf, fmt, fps, nf)


# Return a file name
# Input:
#   cam_id: camera id (int)
#   view_id: view id (int)
#   ds: datetime string (str), "2015-05-22"
#   b: bounding box (dictionary with Left Top Right Bottom), e.g., {"L": 2330, "T": 690, "R": 3730, "B": 2090}
#   w: width (int)
#   h: height (int)
#   sf: starting frame number (int)
#   st: starting epochtime in seconds (int)
#   et: ending epochtime in seconds (int)
def get_file_name(cam_id, view_id, ds, b, w, h, sf, st, et):
    return "%d-%d-%s-%r-%r-%r-%r-%r-%r-%r-%r-%r" % (cam_id, view_id, ds, b["L"], b["T"], b["R"], b["B"], w, h, sf, st, et)


# Convert the camera name to camera ID
def cam_name_to_id(name):
    if name == "clairton1":
        return 0
    elif name == "braddock1":
        return 1
    elif name == "westmifflin1":
        return 2
    else:
        return None


# Get the location of the camera by its ID
def get_cam_location_by_id(cam_id):
    lat = None # latitude
    lng = None # longitude
    if cam_id == 0: # clairton1
        lat = 40.305062
        lng = -79.876692
    elif cam_id == 1: # braddock1
        lat = 40.392967
        lng = -79.855709
    elif cam_id == 2: # westmifflin1
        lat = 40.392967
        lng = -79.855709
    return (lat, lng)


# Given a thumbnail url (having the date, camera, and bounding box information)
# Generate all urls that represents the same day
# Input:
#   url: any thumbnail server url
#   cam_id: the id of the camera (int)
#   view_id: the id of the view (int)
#   video_size: the desired output video size
#   nf: number of frames of each divided video
#   overlap: the number of overlapping frames for each split
# Output:
#   url_part_list: a list of thumbnail url parts for that date, camera, and bounding box
#   file_name_list: a list of file names, corresponding to the url parts
#   ct_sub_list: a list of camera captured times
#   unique_p: an unique path for storing the results
#   uuid: an unique ID for the url
def gen_url_parts(url, cam_id, view_id, video_size=180, nf=36, overlap=18):
    # Get frame captured times
    ds = get_datetime_str_from_url(url)
    cam_name = get_camera_name_from_url(url)
    tm_json = request_json(get_tm_json_url(cam_name=cam_name, ds=ds))
    if tm_json is None:
        print("Error getting frame captured times")
        return None

    # Divide the large video into small ones
    sf_list, sf_dt_list, ef_dt_list, ct_sub_list = divide_start_frame(tm_json["capture-times"], nf=nf, overlap=overlap)
    if sf_list is None:
        print("Error dividing videos")
        return (None, None)

    # Generate the url parts and file names
    b = get_bound_from_url(url)
    url_part_list = []
    file_name_list = []
    for i in range(len(sf_list)):
        sf = sf_list[i]
        url_part_list.append(get_url_part(cam_name=cam_name, ds=ds, b=b, sf=sf, w=video_size, h=video_size, nf=nf))
        st = int(sf_dt_list[i].timestamp())
        et = int(ef_dt_list[i].timestamp())
        file_name_list.append(get_file_name(cam_id, view_id, ds, b, video_size, video_size, sf, st, et))
    unique_p = "%s/%d-%d/" % (ds, cam_id, view_id)
    uuid = "-".join(file_name_list[0].split("-")[:11])
    return (url_part_list, file_name_list, ct_sub_list, unique_p, uuid)


# Call the thumbnail server to generate and get videos
# Then save the videos
# Input:
#   url_part_list: a list of thumbnail url parts for that date, camera, and bounding box
#   file_name_list: a list of file names, corresponding to the url parts
#   url_root: the root of the thumbnail server url
#   vid_p: the folder path to save the videos
#   num_try: the number of times that the function has been called
#   num_workers: the number of workers to download the frames
def download_video(url_part_list, file_name_list, url_root, vid_p, num_try=0, num_workers=8):
    print("="*100)
    print("="*100)
    print("This function has been called for %d times." % num_try)
    if num_try > 30:
        print("Terminate the recursive call due to many errors. Please check manually.")
        return
    num_errors = 0

    # Construct the lists of urls and file paths
    arg_list = []
    for i in range(len(url_part_list)):
        file_p = vid_p + file_name_list[i] + ".mp4"
        url = url_root + url_part_list[i]
        arg_list.append((url, file_p))

    # Download the files in parallel
    result = Pool(num_workers).starmap(urlretrieve_worker, arg_list)
    for r in result:
        if r: num_errors += 1
    if num_errors > 0:
        print("="*60)
        print("Has %d errors. Need to do again." % num_errors)
        num_try += 1
        download_video(url_part_list, file_name_list, url_root, vid_p, num_try=num_try)
    else:
        print("DONE download_video")


# The worker for getting the videos
# Input:
#   url: the url for getting the frames
#   file_p: the path for saving the file
def urlretrieve_worker(url, file_p):
    error = False
    if os.path.isfile(file_p): # skip if the file exists
        print("\t{File exists} %s\n" % file_p)
        return error
    try:
        print("\t{Request} %s\n" % url)
        urllib.request.urlretrieve(url, file_p)
        print("\t{Done} %s\n" % url)
    except Exception as ex:
        print("\t{%s} %s\n" % (ex, url))
        error = True
    return error


# Load the video and convert it to numpy.array
# Then save the numpy.array to a local file
# Input:
# - url: the thumbnail server url
# - vid_p: the path of a folder for loading the video file
# - rgb_p: the path of a folder for saving the rgb frames
# - file_name: the desired file name for the rgb frames (without file extension)
def video_to_numpy(url, vid_p, rgb_p, file_name):
    rgb_vid_in_p = vid_p + file_name + ".mp4"
    rgb_4d_out_p = rgb_p + file_name + ".npy"
    op = OpticalFlow(rgb_vid_in_p=rgb_vid_in_p, rgb_4d_out_p=rgb_4d_out_p, flow_type=None)
    op.process()


# The core function for smoke recognition
# Notice that the input rgb frames has shape (time, height, width, channel) = (n, 180, 180, 3)
# Input:
# - rgb_p: file path that points to the numpy.array file that stores rgb frames
# - file_name_list: list of file names of the numpy.array file
# Output:
# - smoke_pb_list: list of the estimated probabilities of having smoke, with shape (time, probability)
# - smoke_px_list: list of the estimated number of smoke pixels, with shape (time, num_of_smoke_pixels)
def recognize_smoke(rgb_p, file_name_list):
    # Prepare model
    mode = "rgb"
    use_cuda = True
    if use_cuda:
        print("Enable GPU computing...")
    learner = I3dLearner(mode=mode, use_cuda=use_cuda, parallel=False)
    p_model = "../data/saved_i3d/paper_result/full-augm-rgb/55563e4-i3d-rgb-s3/model/573.pt"
    model = learner.set_model(0, 1, mode, p_model, False)
    model.train(False) # set model to evaluate mode (IMPORTANT)
    image_size = 224
    transform = learner.get_transform(mode, image_size=image_size)

    # Iterate
    smoke_pb_list = []
    activation_ratio_list = []
    for fn in file_name_list:
        print("Process file:", fn)
        # Compute probability of having smoke
        print("Estimate the probability of having smoke...")
        v = np.load(rgb_p + fn + ".npy")
        v = transform(v)
        v = torch.unsqueeze(v, 0)
        if use_cuda and torch.cuda.is_available:
            v = v.cuda()
        pred = learner.make_pred(model, v, upsample=False).squeeze().transpose(0, 1)
        pred = F.softmax(pred).cpu().detach().numpy()[:, 1]
        smoke_pb_list.append(list(pred.round(3)))
        # GradCAM (class activation mapping)
        # Compute the ratio of the activated region that will affect the probability
        # This can potentially be used to estimate the number of smoke pixels
        # Need to check more papers about weakly supervised learning
        print("Run GradCAM...")
        grad_cam = GradCam(model, use_cuda=use_cuda)
        target_class = 1 # has smoke
        cam = grad_cam.generate_cam(v, target_class)
        cam = cam.reshape((cam.shape[0], -1))
        #print(pd.DataFrame(data={"cam": cam.flatten()}).describe().applymap(lambda x: "%.4f" % x))
        n_activation_px = np.minimum(np.maximum(cam*2 - 1, 0), 1)
        n_activation_px[n_activation_px>0] = 1
        activation_ratio = list(np.sum(n_activation_px, axis=1, dtype=np.uint32) / (image_size**2))
        activation_ratio_list.append(activation_ratio)
    return (smoke_pb_list, activation_ratio_list)


if __name__ == "__main__":
    main(sys.argv)
