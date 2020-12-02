"""
This script is used to deploy the trained models to recognize smoke emissions
"""

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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


class SmokeEventDataset(Dataset):
    """
    The dataset class for loading RGB numpy files
    """
    def __init__(self, file_name_list, ct_sub_list, root_dir, transform=None):
        """
        Input:
            file_name_list: list of file names of the numpy.array file
            ct_sub_list: the captured time array for each frame (in epochtime format)
            root_dir (string): the root directory that stores video files
            transform (callable, optional): optional transform to be applied on a video
        """
        self.file_name_list = file_name_list
        self.ct_sub_list = ct_sub_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        fn = self.file_name_list[idx] # file name
        epochtime = int(np.max(self.ct_sub_list[idx])) # use the largest timestamp

        # Load video data
        file_path = os.path.join(self.root_dir, fn + ".npy")
        if not is_file_here(file_path):
            raise ValueError("Cannot find file: %s" % (file_path))
        frames = np.load(file_path).astype(np.uint8)

        # Transform video
        if self.transform:
            frames = self.transform(frames)

        frames = torch.unsqueeze(frames, 0)

        # Return item
        return {"frames": frames, "epochtime": epochtime}


def main(argv):
    """
    The main function for smoke recognition (using the trained model)
    """
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
        process_all_urls(nf=nf, use_cuda=True, parallel=True)
        #process_all_urls(nf=nf, use_cuda=True, parallel=True, test_mode=True)
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


def scan_urls(num_workers=8):
    """
    Scan and request all thumbnail server urls

    In this way, the front-end users do not need to wait for the server to render the videos

    Input:
        num_workers: the number of parallel workers to request the urls
    """
    p = "../data/event/" # the path to read the input files
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
    """
    Request an url using urllib.request.urlopen

    Input:
        url: the url to request
    """
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


def check_and_fix_urls():
    """
    Check urls in the json files that will be used to process the probabilities of having smoke

    This function will:
        1. Identify the files that are not named using dates
        2. Identify the urls that do not have the same date as the file name
        3. Fix the urls problems automatically and raise issues for the file name problem
    """
    p = "../data/production_url_list/" # the path to read the input files
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


def process_events(nf=36):
    """
    Process smoke events and save them (in thumbnail server urls)

    Input:
        nf: number of frames of each divided video
    """
    p = "../data/production/" # the path to read the input files
    p_out = "../data/event/" # the path to save the output files
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


def get_event_frame_list(df_event, t_to_f, nf, max_f=None, min_f=36):
    """
    Given the smoke events, get a list of their corresponding frame numbers

    TODO:
        Currently the implementation assumes that events do not overlap
        We need to edit this function to make sure that when event overlaps,
        ...the result is still correct

    Input:
        df_event: the pandas DataFrame that contains epochtime and smoke events
        t_to_f: a dictionary that maps epochtime to frame number
        nf: number of frames of each divided video, e.g., 36
        max_f: max number of frames for each event (this is a soft number that will be affected by min_f)
        min_f: if the remainig event has less than min_f frames, it will be merged

    Output:
        event_frame_list: the list of smoke event starting and ending frame number
    """
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


def get_event_time_list(event_frame_list, f_to_t):
    """
    Given a list of starting and ending frames that represent events,
    ...compute the list of smoke event starting and ending epochtimes and the metadata

    Input:
        event_frame_list: list of starting and ending frames that represent events (from get_event_frame_list function)
        f_to_t: a dictionary that maps frame number to epochtime

    Output:
        event_time_list: the list of smoke event starting and ending epochtimes
        total_event_duration_in_secs: the duration of all the events in seconds
    """
    total_event_duration_in_secs = 0
    event_time_list = []
    for e in event_frame_list:
        start_t = f_to_t[e[0]]
        end_t = f_to_t[e[1]]
        event_time_list.append([start_t, end_t])
        total_event_duration_in_secs += end_t-start_t+1
    return (event_time_list, total_event_duration_in_secs)


def get_smoke_event_urls(event_frame_list, f_to_t, cam_name, ds, b, view_str):
    """
    Given a list of starting and ending frames that represent events,
    ...compute the list of thumbnail server urls for each smoke event

    Input:
        event_frame_list: list of starting and ending frames that represent events
            ...(from get_event_frame_list function)
        f_to_t: a dictionary that maps frame number to epochtime
        cam_name: name of the camera, e.g., "clairton1"
        ds: date string, e.g., "2019-04-02"
        b: bounding box, e.g., {"L": 2330, "T": 690, "R": 3730, "B": 2090}
        view_str: the camera view string, e.g., "0-1"

    Output:
        event_urls: the list of thumbnail server urls for each smoke event
    """
    url_root = "https://thumbnails-v2.createlab.org/thumbnail"
    event_urls = []
    for e in event_frame_list:
        url_part = get_url_part(cam_name=cam_name, ds=ds, b=b, sf=e[0], w=180, h=180, nf=e[1]-e[0]+1, label=True)
        event_urls.append([url_root + url_part, view_str, f_to_t[e[0]], f_to_t[e[1]]])
    return event_urls


def sort_by_index(a, idx, reverse=False):
    """
    Sort an array of array by index

    Input:
        a: a 2D array, e.g., [[1, "a"], [2, "b"]]
        idx: the index to sort the arrays inside the array, e.g., 0
    """
    return sorted(a, key=lambda t: t[idx], reverse=reverse)


def sort_by_camera_view(a, idx):
    """Sort an array by camera view"""
    return sorted(a, key=lambda t: int(t[idx].split("-")[0])*1000+int(t[idx].split("-")[1]))


def sort_camera_view_list(a):
    """Given a list of camera view ID (e.g., "0-0", "0-1"), return a sorted list"""
    return sorted(a, key=lambda t: int(t.split("-")[0])*1000+int(t.split("-")[1]))


def add_smoke_events(esdr_json):
    """
    Given a dictionary in the ESDR file format, compute and add the smoke events

    Here is an example of the ESDR format:
        {"channel_names": ["smoke_probability", "activation_ratio", "event"], "data": [[1546520585, 1.0, 0.067, 0], [1546520675, 1.0, 0.063, 0]]}

    For a record [1546520585, 1.0, 0.067, 0],
    ...1546520585 is the timestamp in epochtime,
    ...1.0 is the probability of having smoke,
    ...0.067 is the activation ratio (defined in the recognize_smoke_worker function)
    ...0 is the event (0 means no smoke, 1 means having smoke)

    Originally the events are all -1,
    ...and this function updates the events
    """
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


def init_data_upload():
    """Register the product on the ESDR system (https://github.com/CMU-CREATE-Lab/esdr)"""
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


def upload_data():
    """Upload smoke recognition results to ESDR system"""
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


def process_all_urls(nf=36, use_cuda=False, parallel=False, test_mode=False):
    """
    Process all thumbnail server urls to recognize smoke emissions

    Input:
        nf: number of frames of each divided video
        use_cuda: use GPU or not
        parallel: use parallel GPU computing or not
        test_mode: run the code for only several days to test if it works
    """
    # Set learner and transform
    learner = I3dLearner(mode="rgb", use_cuda=use_cuda, parallel=parallel)
    if learner.use_cuda:
        print("Enable GPU computing...")
        if torch.cuda.device_count() == 1:
            print("Only one GPU...disable parallel computing...")
            parallel = False
    else:
        print("No GPU or do not want to use GPU...disable parallel computing...")
        parallel = False
    transform = learner.get_transform(learner.mode, image_size=learner.image_size)

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
            flag = process_url(learner, transform, m["url"], m["cam_id"], m["view_id"], nf, parallel, test_mode=test_mode)
            if flag is True and fn not in processed_dates:
                processed_dates.append(date_tr)
        # Save processed dates at each step (to avoid errors in the middle)
        save_json(list(np.unique(processed_dates)), p_dates)


def set_dataloader(rank, world_size, file_name_list, ct_sub_list, root_dir, transform, num_workers, parallel):
    """Set the pytorch dataloader for smoke recognition"""
    print("Set dataloader...")
    dataset = SmokeEventDataset(file_name_list, ct_sub_list, root_dir, transform=transform)
    # We need to set the batch_size to 1 because we want each GPU to process one file at one time (not multiple files)
    if parallel:
        sampler = DistributedSampler(dataset, shuffle=False, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=1,
                num_workers=int(num_workers/world_size), pin_memory=True, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader


def process_url(learner, transform, url, cam_id, view_id, nf, parallel, test_mode=False):
    """
    Process each url and predict the probability of having smoke for that date and view

    Input:
        learner: the Learner object (e.g., I3dLearner)
        transform: the data preprocessing pipeline
        url: the thumbnail server url that we want to process
        cam_id: camera ID
        view_id: view ID
        nf: number of frames of each divided video
        parallel: use GPU parallel computing or not
        test_mode: print extra information or not (for testing)
    """
    print("="*60)
    print("Process %s" % url)
    url_root = "https://thumbnails-v2.createlab.org/thumbnail"

    # Divide the video into several small parts
    url_part_list, file_name_list, ct_sub_list, unique_p, uuid = gen_url_parts(url, cam_id, view_id, nf=nf, overlap=18)
    if url_part_list is None or file_name_list is None:
        print("Error generating url parts...")
        return False

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
    print("Extract video frames...")
    for i in range(len(url_part_list)):
        fn = file_name_list[i]
        url = url_root + url_part_list[i]
        if is_file_here(rgb_p + fn + ".npy"):
            print("Frame file exists...skip...")
        else:
            video_to_numpy(url, vid_p, rgb_p, fn)

    # Apply the smoke recognition model on the video frames
    print("Recognize smoke emissions...")
    smoke_pb_list, activation_ratio_list, epochtime_list = recognize_smoke(learner, transform, rgb_p,
            file_name_list, ct_sub_list, parallel)
    if test_mode:
        print(smoke_pb_list)
        print(activation_ratio_list)
        print(epochtime_list)

    # Put data together for uploading to the ESDR system
    # Notice that for the epochtime, we use the ending time of the video (NOT starting time)
    # The reason is because we want consistent timestamps when doing real-time predictions
    data_json = {"channel_names": ["smoke_probability", "activation_ratio", "event"], "data": []}
    for i in range(len(smoke_pb_list)):
        smoke_pb = smoke_pb_list[i]
        activation_ratio = activation_ratio_list[i]
        epochtime = epochtime_list[i]
        event = -1 # this will be computed later in the add_smoke_events() function
        data_json["data"].append([epochtime, smoke_pb, activation_ratio, event])
    if test_mode:
        print(data_json)
    save_json(data_json, p_root + uuid + ".json")

    print("DONE process_url")
    return True


def get_datetime_str_from_url(url):
    """Parse and get the datetime string from the thumbnail server url"""
    m = re.search("\d+-\d+-\d+\.timemachine", url)
    return m.group(0).split(".")[0]


def get_camera_name_from_url(url):
    """Parse and get the camera name from the thumbnail server url"""
    m = re.search("tiles.cmucreatelab.org/ecam/timemachines/\w+/", url)
    return m.group(0).split("/")[3]


def get_bound_from_url(url):
    """Parse and get the bounding box from the thumbnail server url"""
    b_str = parse_qs(urlparse(url).query)["boundsLTRB"][0]
    b_str_split = list(map(int, map(float, b_str.split(","))))
    return {"L": b_str_split[0], "T": b_str_split[1], "R": b_str_split[2], "B": b_str_split[3]}


def get_tm_json_url(cam_name=None, ds=None):
    """
    Get the url that contains the information about the frame captured times

    Input:
        cam_name: camera name
        ds: date string, e.g., 2019-01-03
    """
    return "https://tiles.cmucreatelab.org/ecam/timemachines/%s/%s.timemachine/tm.json" % (cam_name, ds)


def str_to_time(ds):
    """
    Convert a date string to a datetime object

    Input:
        ds: date string, e.g., 2019-01-03
    """
    return datetime.strptime(ds, "%Y-%m-%d %H:%M:%S")


def divide_start_frame(ct_list, nf=36, overlap=0):
    """
    Given a frame captured time array (obtained from tm.json files, see the get_tm_json_url function)
    ...divide it into a set of starting frames

    Input:
        ct_list: the captured time list
        nf: number of frames of each divided video
        overlap: the number of overlapping frames for each split

    Output:
        sf_list: the starting frame number
        sf_dt_list: the starting datetime
        ef_dt_list: the ending datetime
        ct_sub_list: the captured time array for each frame (in epochtime format)
    """
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
    L = len(ct_list)
    for sf in r:
        ef = sf + nf - 1 # end frame
        if ef > frame_max or ef >= L: break
        sf_list.append(sf)
        sf_dt_list.append(str_to_time(ct_list[sf]))
        ef_dt_list.append(str_to_time(ct_list[ef]))
        ct_item = []
        for i in range(sf, sf + nf):
            ct_item.append(str_to_time(ct_list[i]).timestamp())
        ct_item = list(map(int, ct_item))
        ct_sub_list.append(ct_item)
    return (sf_list, sf_dt_list, ef_dt_list, ct_sub_list)


def get_url_part(cam_name=None, ds=None, b=None, w=180, h=180, sf=None, fmt="mp4", fps=12, nf=36, label=False):
    """
    Return a thumbnail server url part

    Input:
        cam_name: camera name (str), e.g., "clairton1"
        ds: datetime string (str), "2015-05-22"
        b: bounding box (dictionary with Left Top Right Bottom), e.g., {"L": 2330, "T": 690, "R": 3730, "B": 2090}
        w: width (int)
        h: height (int)
        sf: starting frame number (int)
        fmt: format (str), "gif" or "mp4" or "png"
        fps: frames per second (int)
        nf: number of frames (int)
        label: add timestamp label on the video or not
    """
    url_part = "?root=http://tiles.cmucreatelab.org/ecam/timemachines/%s/%s.timemachine/&boundsLTRB=%r,%r,%r,%r&width=%r&height=%r&startFrame=%r&format=%s&fps=%r&tileFormat=mp4&nframes=%r" % (cam_name, ds, b["L"], b["T"], b["R"], b["B"], w, h, sf, fmt, fps, nf)
    if label:
        url_part += "&labelsFromDataset"
    return url_part


def get_file_name(cam_id, view_id, ds, b, w, h, sf, st, et):
    """
    Return a file name

    Input:
        cam_id: camera id (int)
        view_id: view id (int)
        ds: datetime string (str), "2015-05-22"
        b: bounding box (dictionary with Left Top Right Bottom), e.g., {"L": 2330, "T": 690, "R": 3730, "B": 2090}
        w: width (int)
        h: height (int)
        sf: starting frame number (int)
        st: starting epochtime in seconds (int)
        et: ending epochtime in seconds (int)
    """
    return "%d-%d-%s-%r-%r-%r-%r-%r-%r-%r-%r-%r" % (cam_id, view_id, ds, b["L"], b["T"], b["R"], b["B"], w, h, sf, st, et)


def cam_id_to_name(cam_id):
    """Convert the camera ID to camera name"""
    if cam_id == 0:
        return "clairton1"
    elif cam_id == 1:
        return "braddock1"
    elif cam_id == 2:
        return "westmifflin1"
    else:
        return None


def cam_name_to_id(name):
    """Convert the camera name to camera ID"""
    if name == "clairton1":
        return 0
    elif name == "braddock1":
        return 1
    elif name == "westmifflin1":
        return 2
    else:
        return None


def get_cam_location_by_id(cam_id):
    """Get the location of the camera by its ID"""
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


def gen_url_parts(url, cam_id, view_id, video_size=180, nf=36, overlap=18):
    """
    Given a thumbnail url (having the date, camera, and bounding box information),
    ...generate all urls that represents the same day

    Input:
        url: any thumbnail server url
        cam_id: the id of the camera (int)
        view_id: the id of the view (int)
        video_size: the desired output video size
        nf: number of frames of each divided video
        overlap: the number of overlapping frames for each split

    Output:
        url_part_list: a list of thumbnail url parts for that date, camera, and bounding box
        file_name_list: a list of file names, corresponding to the url parts
        ct_sub_list: a list of camera captured times (in epochtime format)
        unique_p: an unique path for storing the results
        uuid: an unique ID for the url
    """
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


def download_video(url_part_list, file_name_list, url_root, vid_p, num_try=0, num_workers=8):
    """
    Call the thumbnail server to generate and get videos, and then save the videos

    Input:
        url_part_list: a list of thumbnail url parts for that date, camera, and bounding box
        file_name_list: a list of file names, corresponding to the url parts
        url_root: the root of the thumbnail server url
        vid_p: the folder path to save the videos
        num_try: the number of times that the function has been called
        num_workers: the number of workers to download the frames
    """
    print("-"*60)
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


def urlretrieve_worker(url, file_p):
    """
    The worker for getting the videos

    Input:
        url: the url for getting the frames
        file_p: the path for saving the file
    """
    error = False
    if os.path.isfile(file_p): # skip if the file exists
        print("{Exist} %s\n" % file_p)
        return error
    try:
        print("{Request} %s\n" % url)
        urllib.request.urlretrieve(url, file_p)
        print("{Done} %s\n" % url)
    except Exception as ex:
        print("{%s} %s\n" % (ex, url))
        error = True
    return error


def video_to_numpy(url, vid_p, rgb_p, file_name):
    """
    Load the video and convert it to numpy.array
    ...and then save the numpy.array to a local file

    Input:
        url: the thumbnail server url
        vid_p: the path of a folder for loading the video file
        rgb_p: the path of a folder for saving the rgb frames
        file_name: the desired file name for the rgb frames (without file extension)
    """
    rgb_vid_in_p = vid_p + file_name + ".mp4"
    rgb_4d_out_p = rgb_p + file_name + ".npy"
    op = OpticalFlow(rgb_vid_in_p=rgb_vid_in_p, rgb_4d_out_p=rgb_4d_out_p, flow_type=None)
    op.process()


def recognize_smoke(learner, transform, rgb_p, file_name_list, ct_sub_list, parallel, smoke_thr=0.6, activation_thr=0.85):
    """
    Perform parallel computing when recognizing smoke

    Input: (see the docstring in the recognize_smoke_worker function)
    """
    # Spawn processes
    if parallel:
        n_gpu = torch.cuda.device_count()
        print("Let's use " + str(n_gpu) + " GPUs!")
        queue = mp.get_context("spawn").Queue()
        mp.spawn(recognize_smoke_worker, nprocs=n_gpu,
                args=(n_gpu, learner, transform, rgb_p, file_name_list, ct_sub_list,
                    parallel, smoke_thr, activation_thr, queue), join=True)
        smoke_pb_list, activation_ratio_list, epochtime_list = [], [], []
        while not queue.empty():
            x = queue.get()
            smoke_pb_list += x[0]
            activation_ratio_list += x[1]
            epochtime_list += x[2]
            del x
    else:
        smoke_pb_list, activation_ratio_list, epochtime_list = recognize_smoke_worker(0, 1, learner, transform, rgb_p,
                file_name_list, ct_sub_list, parallel, smoke_thr, activation_thr, None)
    return (smoke_pb_list, activation_ratio_list, epochtime_list)


def recognize_smoke_worker(rank, world_size, learner, transform, rgb_p,
        file_name_list, ct_sub_list, parallel, smoke_thr, activation_thr, queue):
    """
    The core function for smoke recognition

    Notice that the input rgb frames has shape (time, height, width, channel) = (n, 180, 180, 3)

    Input:
        rank: unique ID for the process (e.g., 0)
        world_size: total number of processes on a machine (e.g., 4)
        learner: the Learner object (e.g., I3dLearner)
        transform: the data pre-processing pipeline
        rgb_p: file path that points to the numpy.array file that stores rgb frames
        file_name_list: list of file names of the numpy.array file
        ct_sub_list: a list of capture times (in epochtime format)
        parallel: use GPU parallel computing or not
        smoke_thr: the threshold (probability between 0 and 1) to determine if smoke exists (e.g., 0.6)
        activation_thr: the threshold (ratio between 0 and 1) to determine if a pixel is activated by GradCAM (e.g., 0.85)
        queue: the shared memory for returing data (when in the parallel mode using multiple GPUs)

    Output:
        smoke_pb_list: list of the estimated probabilities of having smoke, with shape (time, number_of_files)
        activation_ratio_list: list of GradCAM activation ratios, with shape (time, number_of_files)
        epochtime_list: list of epochtime for the resulting video clip (using the largest timestamp)
    """
    # Set the dataloader
    num_workers = max(mp.cpu_count()-2, 0)
    dataloader = set_dataloader(rank, world_size, file_name_list, ct_sub_list, rgb_p, transform, num_workers, parallel)

    # Set model
    p_model = "../data/saved_i3d/paper_result/full-augm-rgb/55563e4-i3d-rgb-s3/model/573.pt"
    model = learner.set_model(rank, world_size, learner.mode, p_model, parallel, phase="test")
    model.train(False) # set model to evaluate mode (IMPORTANT)
    grad_cam = GradCam(model, use_cuda=learner.use_cuda, normalize=False)

    # Iterate over batch data
    smoke_pb_list = []
    activation_ratio_list = []
    epochtime_list = []
    for d in tqdm.tqdm(dataloader):
        epochtime_list.append(int(d["epochtime"][0]))
        # Compute probability of having smoke
        v = d["frames"][0]
        if learner.use_cuda and torch.cuda.is_available:
            v = v.cuda()
        pred, pred_upsample = learner.make_pred(model, v, upsample=None)
        pred = F.softmax(pred.squeeze().transpose(0, 1)).cpu().detach().numpy()[:, 1]
        pred_upsample = F.softmax(pred_upsample.squeeze().transpose(0, 1)).cpu().detach().numpy()[:, 1]
        smoke_pb = np.median(pred) # use the median as the probability
        smoke_pb_list.append(round(float(smoke_pb), 3))
        # GradCAM (class activation mapping)
        # Compute the ratio of the activated region that will affect the probability
        # This can potentially be used to estimate the number of smoke pixels
        # Need to check more papers about weakly supervised learning
        C = grad_cam.generate_cam(v, 1) # 1 is the target class, which means having smoke emissions
        C = C.reshape((C.shape[0], -1))
        #print(pd.DataFrame(data={"GradCAM": C.flatten()}).describe().applymap(lambda x: "%.3f" % x))
        if smoke_pb > smoke_thr: # only compute the activation ratio when smoke is predicted
            C = np.multiply(C > activation_thr, 1) # make the binary mask
            activation_ratio = np.sum(C, axis=1, dtype=np.uint32) / (learner.image_size**2)
            activation_ratio[pred_upsample < smoke_thr] = 0
            activation_ratio = np.mean(activation_ratio) # use the mean as the activation ratio
            activation_ratio_list.append(round(float(activation_ratio), 3))
        else:
            activation_ratio_list.append(0.0)

    if queue is None:
        return (smoke_pb_list, activation_ratio_list, epochtime_list)
    else:
        queue.put((smoke_pb_list, activation_ratio_list, epochtime_list))


if __name__ == "__main__":
    main(sys.argv)
