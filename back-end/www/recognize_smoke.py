import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # use the order in the nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # specify which GPU(s) to be used
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
from collections import OrderedDict
from collections import defaultdict


# The main function for smoke recognition (using the trained model)
def main(argv):
    if len(argv) < 2:
        print("Usage:")
        print("python recognize_smoke.py check_and_fix_urls")
        print("python recognize_smoke.py process_all_urls")
        print("python recognize_smoke.py process_events")
        print("python recognize_smoke.py init_data_upload")
        print("python recognize_smoke.py upload_data")
        print("python recognize_smoke.py scan_urls")
        return

    program_start_time = time.time()

    # Parameters
    nf = 36 # number of frames of each divided video

    if argv[1] == "check_and_fix_urls":
        check_and_fix_urls()
    elif argv[1] == "process_all_urls":
        #process_all_urls(nf=nf)
        process_all_urls(nf=nf, test_mode=True)
    elif argv[1] == "process_events":
        process_events(nf=nf)
    elif argv[1] == "init_data_upload":
        init_data_upload()
    elif argv[1] == "upload_data":
        upload_data()
    elif argv[1] == "scan_urls":
        scan_urls()
    else:
        print("Wrong usage. Run 'python recognize_smoke.py' for details.")

    program_run_time = (time.time()-program_start_time)/60
    print("Took %.2f minutes to run the program" % program_run_time)
    print("END")


# Scan and request all thumbnail server urls
# So that the front-end users do not need to wait for generating the videos
def scan_urls(num_workers=8):
    p = "../data/event/"
    event_metadata = load_json(p + "event_metadata.json")
    pool = Pool(num_workers)
    for date_str in event_metadata:
        event_json = load_json(p + date_str + ".json")
        url_list = []
        for cam_id in event_json:
            for view_id in event_json[cam_id]["url"]:
                url_list += event_json[cam_id]["url"][view_id]["url"]
        pool.starmap(url_open_worker, url_list)
    pool.close()
    pool.join()


def url_open_worker(url, *args):
    url = url.replace("width=180", "width=320").replace("height=180", "height=320")
    try:
        response = urllib.request.urlopen(url)
        print("Response %d for %s" % (response.getcode(), url))
    except urllib.error.HTTPError as e:
        print("ERROR when opening url ---- %s" % url)
        print("Cannot fulfill the request with error code: ", e.code)
    except urllib.error.URLError as e:
        print("ERROR when opening url ---- %s" % url)
        print("Failed to reach a server with reason: ", e.reason)


# Check urls in the json files in a folder
# Identify the files that are not named using dates
# Identify the urls that do not have the same date as the file name
# Fix the urls problems automatically and raise issues for the file name problem
def check_and_fix_urls():
    p = "../data/production_url_list/"
    for fn in get_all_file_names_in_folder(p):
        if ".json" not in fn: continue
        date_in_fn = re.findall(r"[\d]{4}-[\d]{2}-[\d]{2}", fn)
        if len(date_in_fn) != 1:
            print("ERROR: file name is not a date, need to fix '%s' manually" % fn)
            continue
        date_in_fn = date_in_fn[0]
        urls = load_json(p + fn)
        has_problem = False
        for i in range(len(urls)):
            date_in_url = get_datetime_str_from_url(urls[i]["url"])
            if date_in_fn != date_in_url:
                print("PROBLEM: date mismatch (file name has %s but url has %s)" % (date_in_fn, date_in_url))
                has_problem = True
                print("Fix the date mismatch problem automatically...")
                urls[i]["url"] = urls[i]["url"].replace(date_in_url, date_in_fn)
        if has_problem:
            print("Replace file with the fixed version: %s" % fn)
            save_json(urls, p + fn)


# Process smoke events and save them (in thumbnail server urls)
# Input:
#   nf: number of frames of each divided video
def process_events(nf=36):
    p = "../data/production/"
    p_out = "../data/event/"
    check_and_create_dir(p_out)
    if is_file_here(p_out + "event_metadata.json"):
        e_metadata = load_json(p_out + "event_metadata.json")
    else:
        e_metadata = {}
    processed_dates = [fn.split(".")[0] for fn in get_all_file_names_in_folder(p_out)]
    for ds in get_all_dir_names_in_folder(p): # date string
        if ds in processed_dates:
            # Ignore dates that are processed
            print("Date %s was processed...skip..." % ds)
            continue
        print("Process date %s" % ds)
        event_json = defaultdict(lambda: {"url": defaultdict(dict)})
        t_to_f = {}
        f_to_t = {}
        e_metadata[ds] = {"cam_list": defaultdict(dict)}
        df_events = defaultdict(list) # collect events, group them by camera id
        for vn in get_all_dir_names_in_folder(p + ds + "/"): # camera view ID
            print("\tProcess view %s" % vn)
            cam_id = int(vn.split("-")[0])
            cam_name = cam_id_to_name(cam_id)
            # Construct the dictionary that maps epochtime to frame number, and also the reverse mapping
            if cam_id not in t_to_f:
                t_to_f[cam_id] = {}
                f_to_t[cam_id] = {}
                tm_json = request_json(get_tm_json_url(cam_name=cam_name, ds=ds))
                ct = tm_json["capture-times"]
                for i in range(len(ct)):
                    t = int(str_to_time(ct[i]).timestamp())
                    t_to_f[cam_id][t] = i
                    f_to_t[cam_id][i] = t
            # Parse the ESDR json file
            for fn in get_all_file_names_in_folder(p + ds + "/" + vn + "/"): # json file
                if ".json" not in fn: continue
                print("\t\tProcess file %s" % fn)
                s = fn.split("-")
                b = {"L": int(s[5]), "T": int(s[6]), "R": int(s[7]), "B": int(s[8])}
                fp = p + ds + "/" + vn + "/" + fn
                esdr_json = load_json(fp)
                esdr_json = add_smoke_events(esdr_json)
                df_event = pd.DataFrame(data=esdr_json["data"], columns=["epochtime"]+esdr_json["channel_names"])
                df_event = df_event.sort_values(by=["epochtime"]).reset_index(drop=True)
                e_frame = get_event_frame_list(df_event, t_to_f[cam_id], nf, max_f=72, min_f=36)
                event_urls = get_smoke_event_urls(e_frame, f_to_t[cam_id], cam_name, ds, b, vn)
                e_times, e_duration = get_event_time_list(e_frame, f_to_t[cam_id])
                df_events[cam_id].append(df_event)
                event_json[cam_id]["url"][vn]["url"] = event_urls
                event_json[cam_id]["url"][vn]["event"] = e_times
                event_json[cam_id]["url"][vn]["total_event_duration_in_secs"] = e_duration
                save_json(esdr_json, fp)
        # Merge events for all camera ids and compute the event list
        for cam_id in df_events:
            df_aggr_event = df_events[cam_id][0]["event"]
            for df in df_events[cam_id][1:]:
                df_aggr_event = df_aggr_event | df["event"]
            df_aggr_event = df_aggr_event.to_frame()
            df_aggr_event["epochtime"] = df_events[cam_id][0]["epochtime"]
            e_frame = get_event_frame_list(df_aggr_event, t_to_f[cam_id], nf, max_f=72, min_f=36)
            e_times, e_duration = get_event_time_list(e_frame, f_to_t[cam_id])
            event_json[cam_id]["event"] = e_times
            event_json[cam_id]["total_event_duration_in_secs"] = e_duration
            e_metadata[ds]["cam_list"][cam_id]["total_event_duration_in_secs"] = e_duration
        # Sort items by camera id
        for cam_id in event_json:
            event_json[cam_id]["url"] = OrderedDict(sort_by_camera_view(event_json[cam_id]["url"].items(), 0))
        # Save the events by camera view
        save_json(event_json, p_out + ds + ".json")
    # Save the date list
    e_metadata = OrderedDict(sorted(e_metadata.items())) # sort by date
    save_json(e_metadata, p_out + "event_metadata.json")


# Given the smoke events, get a list of their corresponding frame numbers
# TODO: currently the implementation assumes that events do not overlap
# TODO: we need to edit this function to make sure that when event overlaps, the result is still correct
# Input:
#   df_event: the pandas DataFrame that contains epochtime and smoke events
#   t_to_f: a dictionary that maps epochtime to frame number
#   nf: number of frames of each divided video, e.g., 36
#   max_f: max number of frames for each event (this is a soft number that will be affected by min_f)
#   min_f: if the remainig event has less than min_f frames, it will be merged
# Output:
#   event_frame_list: the list of smoke event starting and ending frame number
def get_event_frame_list(df_event, t_to_f, nf, max_f=None, min_f=36):
    event_idx_list = array_to_event(list(df_event["event"])) # a list of starting and ending indices of the events
    epochtime = df_event["epochtime"]
    event_frame_list = []
    for e in event_idx_list:
        i, j = e[0], e[1]
        start_f = t_to_f[epochtime[i]] - nf
        end_f = t_to_f[epochtime[j]]
        need_merge = False
        if max_f is not None:
            L = end_f - start_f + 1
            if L > max_f and L % max_f < min_f:
                need_merge = True
            while end_f-start_f+1 > max_f:
                event_frame_list.append([start_f, start_f+max_f-1])
                start_f += max_f
        if need_merge:
            event_frame_list[-1][1] = end_f
        else:
            event_frame_list.append([start_f, end_f])
    return event_frame_list


# Given a list of starting and ending frames that represent events
# Compute the list of smoke event starting and ending epochtimes and the metadata
# Input:
#   event_frame_list: list of starting and ending frames that represent events (from get_event_frame_list function)
#   f_to_t: a dictionary that maps frame number to epochtime
# Output:
#   event_time_list: the list of smoke event starting and ending epochtimes
#   total_event_duration_in_secs: the duration of all the events in seconds
def get_event_time_list(event_frame_list, f_to_t):
    total_event_duration_in_secs = 0
    event_time_list = []
    for e in event_frame_list:
        start_t = f_to_t[e[0]]
        end_t = f_to_t[e[1]]
        event_time_list.append([start_t, end_t])
        total_event_duration_in_secs += end_t-start_t+1
    return (event_time_list, total_event_duration_in_secs)


# Given a list of starting and ending frames that represent events
# Compute the list of thumbnail server urls for each smoke event
# Input:
#   event_frame_list: list of starting and ending frames that represent events (from get_event_frame_list function)
#   f_to_t: a dictionary that maps frame number to epochtime
#   cam_name: name of the camera, e.g., "clairton1"
#   ds: date string, e.g., "2019-04-02"
#   b: bounding box, e.g., {"L": 2330, "T": 690, "R": 3730, "B": 2090}
#   view_str: the camera view string, e.g., "0-1"
# Output:
#   event_urls: the list of thumbnail server urls for each smoke event
def get_smoke_event_urls(event_frame_list, f_to_t, cam_name, ds, b, view_str):
    url_root = "https://thumbnails-v2.createlab.org/thumbnail"
    event_urls = []
    for e in event_frame_list:
        url_part = get_url_part(cam_name=cam_name, ds=ds, b=b, sf=e[0], w=180, h=180, nf=e[1]-e[0]+1, label=True)
        event_urls.append([url_root + url_part, view_str, f_to_t[e[0]], f_to_t[e[1]]])
    return event_urls


# Sort an array by number
def sort_by_number(a, idx, reverse=False):
    return sorted(a, key=lambda t: t[idx], reverse=reverse)


# Sort an array by camera view
def sort_by_camera_view(a, idx):
    return sorted(a, key=lambda t: int(t[idx].split("-")[0])*1000+int(t[idx].split("-")[1]))


# Given a list of camera view ID (e.g., "0-0", "0-1"), return a sorted list
def sort_camera_view_list(a):
    return sorted(a, key=lambda t: int(t.split("-")[0])*1000+int(t.split("-")[1]))


# Given an esdr json, compute and add the smoke events
def add_smoke_events(esdr_json):
    data = esdr_json["data"]
    max_event_gap_count = 1 # the max number of the gaps to merge events
    smoke_pb_thr = 0.85 # the probability threshold to define a smoke event
    activation_ratio_thr = 0.4 # the activation ratio threshold to define a smoke event
    idx_to_fill = None # the index list to fill the event gaps
    for i in range(len(data)):
        smoke_pb = data[i][1]
        activation_ratio = data[i][2]
        event = 1 if smoke_pb > smoke_pb_thr and activation_ratio > activation_ratio_thr else 0
        esdr_json["data"][i][3] = event
        # Fill the event gap
        if event == 1:
            if idx_to_fill is not None and len(idx_to_fill) <= max_event_gap_count:
                for j in idx_to_fill:
                    esdr_json["data"][j][3] = 1 # fill the gaps
            idx_to_fill = []
        else:
            if idx_to_fill is not None:
                idx_to_fill.append(i)
    return esdr_json


# Register the product on the ESDR system
def init_data_upload():
    # Specify the data format (the definition of "product" on ESDR)
    product_json = {
      "name": "RISE_smoke_recognition_v3",
      "prettyName": "Recognizing Industrial Smoke Emissions",
      "vendor": "CMU CREATE Lab",
      "description": "Recognizing Industrial Smoke Emissions",
      "defaultChannelSpecs": {
        "version": 1,
        "channels": {
          "smoke_probability": {
            "prettyName": "the probability of having smoke",
            "units": "probability",
            "range": {
              "min": 0,
              "max": 1
            }
          },
          "activation_ratio": {
            "prettyName": "the ratio of activation region",
            "units": "ratio",
            "range": {
              "min": 0,
              "max": 1
            }
          },
          "event": {
            "prettyName": "the smoke event",
            "units": "no/yes",
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
    product_id = 97 # this ID is for production

    # Get the access token
    access_token, _ = get_esdr_access_token(load_json("../data/auth.json"))
    if access_token is None:
        print("ERROR! No access token.")
        return

    # Upload all data
    p = "../data/production/"
    for ds in get_all_dir_names_in_folder(p): # date string
        for vn in get_all_dir_names_in_folder(p + ds + "/"): # camera view ID
            for fn in get_all_file_names_in_folder(p + ds + "/" + vn + "/"): # json file
                if ".json" not in fn: continue
                data = load_json(p + ds + "/" + vn + "/" + fn)
                if "channel_names" not in data or "data" not in data: continue
                s = vn.split("-")
                lat, lng = get_cam_location_by_id(int(s[0]))
                name = "RISE_smoke_recognition_v3_camera_%s_view_%s" % (s[0], s[1])
                upload_data_to_esdr(name, data, product_id, access_token, isPublic=1, latitude=lat, longitude=lng)


# Process all thumbnail server urls
# Input:
#   nf: number of frames of each divided video
def process_all_urls(nf=36, test_mode=False):
    # Set models
    learner, grad_cam, transform = set_grad_cam_model()

    # Read processed dates
    p_dates = "../data/production/processed_dates.json"
    if is_file_here(p_dates):
        processed_dates = load_json(p_dates)
    else:
        processed_dates = []

    # Process dates
    p = "../data/production_url_list/"
    for fn in get_all_file_names_in_folder(p):
        if ".json" not in fn: continue
        date_tr = fn.split(".json")[0]
        if date_tr in processed_dates:
            print("Date %s is already processed...skip..." % date_tr)
            continue
        if test_mode and "2019-02-03" not in fn and "2019-02-04" not in fn: continue
        m = load_json(p + fn)
        c = 0
        for m in load_json(p + fn):
            c += 1
            if test_mode and c > 2: break
            if "url" not in m or "cam_id" not in m or "view_id" not in m: continue
            flag = process_url(learner, grad_cam, transform, m["url"], m["cam_id"], m["view_id"], nf=nf, test_mode=test_mode)
            if flag is True and fn not in processed_dates:
                processed_dates.append(date_tr)
        # Save processed dates at each step (to avoid errors in the middle)
        save_json(list(np.unique(processed_dates)), p_dates)


# Set the model for processing GradCAM
# Input:
#   use_cuda: use cuda (GPU computing) or not
#   parallel: use nn.DistributedDataParallel or not
# Output:
#   learner: the Learner object (e.g., I3dLearner)
#   grad_cam: the GradCam object
#   transform: the data preprocessing pipeline
def set_grad_cam_model(use_cuda=True, parallel=False):
    # Prepare learner and model
    if use_cuda:
        print("Enable GPU computing...")
    mode = "rgb"
    learner = I3dLearner(mode=mode, use_cuda=use_cuda, parallel=parallel)
    p_model = "../data/saved_i3d/paper_result/full-augm-rgb/55563e4-i3d-rgb-s3/model/573.pt"
    model = learner.set_model(0, 1, mode, p_model, parallel, phase="test")
    model.train(False) # set model to evaluate mode (IMPORTANT)
    transform = learner.get_transform(mode, image_size=learner.image_size)
    grad_cam = GradCam(model, use_cuda=use_cuda, normalize=False)
    return (learner, grad_cam, transform)


# Process each url and predict the probability of having smoke for that date and view
# Input:
#   learner: the Learner object (e.g., I3dLearner)
#   grad_cam: the GradCam object
#   transform: the data preprocessing pipeline
#   url: the thumbnail server url that we want to process
#   cam_id: camera ID
#   view_id: view ID
#   nf: number of frames of each divided video
def process_url(learner, grad_cam, transform, url, cam_id, view_id, nf=36, test_mode=False):
    print("Process %s" % url)
    url_root = "https://thumbnails-v2.createlab.org/thumbnail"

    # Divide the video into several small parts
    url_part_list, file_name_list, ct_sub_list, unique_p, uuid = gen_url_parts(url, cam_id, view_id, nf=nf, overlap=18)
    if url_part_list is None or file_name_list is None:
        print("Error generating url parts...")
        return False

    # For testing only
    #url_part_list = url_part_list[:10]
    #file_name_list = file_name_list[:10]
    #ct_sub_list = ct_sub_list[:10]

    # The directory for saving the files
    p_root = "../data/production/" + unique_p
    vid_p = p_root + "video/"
    rgb_p = p_root + "rgb/"
    check_and_create_dir(rgb_p)
    check_and_create_dir(vid_p)

    # Skip if the json file exists
    if is_file_here(p_root + uuid + ".json"):
        print("File %s exists...skip..." % (uuid + ".json"))
        return True

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
    smoke_pb_list, activation_ratio_list = recognize_smoke(learner, grad_cam, transform,
            rgb_p, file_name_list, test_mode=test_mode)
    if test_mode:
        print(smoke_pb_list)
        print(activation_ratio_list)

    # Put data together for uploading to the ESDR system
    # Notice that for the epochtime, we use the ending time of the video (NOT starting time)
    # The reason is because we want consistent timestamps when doing real-time predictions
    data_json = {"channel_names": ["smoke_probability", "activation_ratio", "event"], "data": []}
    for i in range(len(smoke_pb_list)):
        smoke_pb = smoke_pb_list[i]
        activation_ratio = activation_ratio_list[i]
        ct_sub = ct_sub_list[i]
        epochtime = int(np.max(ct_sub)) # use the largest timestamp
        event = -1 # this will be computed later in the add_smoke_events() function
        data_item = [epochtime, smoke_pb, activation_ratio, event]
        data_json["data"].append(data_item)
    if test_mode:
        print(data_json)
    save_json(data_json, p_root + uuid + ".json")

    print("DONE process_url")
    return True


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
    b_str_split = list(map(int, map(float, b_str.split(","))))
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
#   label: add timestamp label on the video or not
def get_url_part(cam_name=None, ds=None, b=None, w=180, h=180, sf=None, fmt="mp4", fps=12, nf=36, label=False):
    url_part = "?root=http://tiles.cmucreatelab.org/ecam/timemachines/%s/%s.timemachine/&boundsLTRB=%r,%r,%r,%r&width=%r&height=%r&startFrame=%r&format=%s&fps=%r&tileFormat=mp4&nframes=%r" % (cam_name, ds, b["L"], b["T"], b["R"], b["B"], w, h, sf, fmt, fps, nf)
    if label:
        url_part += "&labelsFromDataset"
    return url_part


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


# Convert the camera ID to camera name
def cam_id_to_name(cam_id):
    if cam_id == 0:
        return "clairton1"
    elif cam_id == 1:
        return "braddock1"
    elif cam_id == 2:
        return "westmifflin1"
    else:
        return None


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
        return (None, None, None, None, None)

    # Divide the large video into small ones
    sf_list, sf_dt_list, ef_dt_list, ct_sub_list = divide_start_frame(tm_json["capture-times"], nf=nf, overlap=overlap)
    if sf_list is None:
        print("Error dividing videos")
        return (None, None, None, None, None)

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
    pool = Pool(num_workers)
    result = pool.starmap(urlretrieve_worker, arg_list)
    pool.close()
    pool.join()
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
#   url: the thumbnail server url
#   vid_p: the path of a folder for loading the video file
#   rgb_p: the path of a folder for saving the rgb frames
#   file_name: the desired file name for the rgb frames (without file extension)
def video_to_numpy(url, vid_p, rgb_p, file_name):
    rgb_vid_in_p = vid_p + file_name + ".mp4"
    rgb_4d_out_p = rgb_p + file_name + ".npy"
    op = OpticalFlow(rgb_vid_in_p=rgb_vid_in_p, rgb_4d_out_p=rgb_4d_out_p, flow_type=None)
    op.process()


# The core function for smoke recognition
# Notice that the input rgb frames has shape (time, height, width, channel) = (n, 180, 180, 3)
# Input:
#   learner: the Learner object (e.g., I3dLearner)
#   grad_cam: the GradCam object
#   transform: the data pre-processing pipeline
#   rgb_p: file path that points to the numpy.array file that stores rgb frames
#   file_name_list: list of file names of the numpy.array file
#   smoke_thr: the threshold (probability between 0 and 1) to determine if smoke exists (e.g., 0.6)
#   activation_thr: the threshold (ratio between 0 and 1) to determine if a pixel is activated by GradCAM (e.g., 0.85)
# Output:
#   smoke_pb_list: list of the estimated probabilities of having smoke, with shape (time, number_of_files)
#   activation_ratio_list: list of GradCAM activation ratios, with shape (time, number_of_files)
def recognize_smoke(learner, grad_cam, transform, rgb_p, file_name_list, test_mode=False, smoke_thr=0.6, activation_thr=0.85):
    smoke_pb_list = []
    activation_ratio_list = []
    for fn in file_name_list:
        print("Process file:", fn)
        # Compute probability of having smoke
        print("Estimate the probability of having smoke...")
        v = np.load(rgb_p + fn + ".npy")
        v = transform(v)
        v = torch.unsqueeze(v, 0)
        if learner.use_cuda and torch.cuda.is_available:
            v = v.cuda()
        pred, pred_upsample = learner.make_pred(grad_cam.model, v, upsample=None)
        pred = F.softmax(pred.squeeze().transpose(0, 1)).cpu().detach().numpy()[:, 1]
        pred_upsample = F.softmax(pred_upsample.squeeze().transpose(0, 1)).cpu().detach().numpy()[:, 1]
        smoke_pb = np.median(pred) # use the median as the probability
        smoke_pb_list.append(round(float(smoke_pb), 3))
        # GradCAM (class activation mapping)
        # Compute the ratio of the activated region that will affect the probability
        # This can potentially be used to estimate the number of smoke pixels
        # Need to check more papers about weakly supervised learning
        print("Run GradCAM...")
        C = grad_cam.generate_cam(v, 1) # 1 is the target class, which means having smoke emissions
        C = C.reshape((C.shape[0], -1))
        #if test_mode: print(pd.DataFrame(data={"GradCAM": C.flatten()}).describe().applymap(lambda x: "%.3f" % x))
        if smoke_pb > smoke_thr: # only compute the activation ratio when smoke is predicted
            C = np.multiply(C > activation_thr, 1) # make the binary mask
            activation_ratio = np.sum(C, axis=1, dtype=np.uint32) / (learner.image_size**2)
            activation_ratio[pred_upsample < smoke_thr] = 0
            activation_ratio = np.mean(activation_ratio) # use the mean as the activation ratio
            activation_ratio_list.append(round(float(activation_ratio), 3))
        else:
            activation_ratio_list.append(0.0)
    return (smoke_pb_list, activation_ratio_list)


if __name__ == "__main__":
    main(sys.argv)
