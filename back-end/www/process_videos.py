import os
import sys
from util import *
import numpy as np
from optical_flow.optical_flow import OpticalFlow
from multiprocessing import Pool

thread = "1"
os.environ["MKL_NUM_THREADS"] = thread
os.environ["NUMEXPR_NUM_THREADS"] = thread
os.environ["OMP_NUM_THREADS"] = thread
os.environ["VECLIB_MAXIMUM_THREADS"] = thread
os.environ["OPENBLAS_NUM_THREADS"] = thread
import cv2 as cv
cv.setNumThreads(12)

# Process videos into rgb frame files and optical flow files
# The file format is numpy.array
def main(argv):
    rgb_dir = "../data/rgb/"
    flow_dir = "../data/flow/"
    metadata_path = "../data/metadata.json"
    num_workers = 1

    # Check for saving directories and create if they don't exist
    check_and_create_dir(rgb_dir)
    check_and_create_dir(flow_dir)

    metadata = load_json(metadata_path)
    p = Pool(num_workers)
    p.map(compute_and_save_flow, metadata)
    print("Done process_videos.py")


def compute_and_save_flow(video_data):
    video_dir = "../data/videos/"
    rgb_dir = "../data/rgb/"
    flow_dir = "../data/flow/"
    file_name = video_data["file_name"]
    rgb_vid_in_p = str(video_dir + file_name + ".mp4")
    rgb_4d_out_p = str(rgb_dir + file_name + ".npy")
    flow_4d_out_p = str(flow_dir + file_name + ".npy")
    if not is_file_here(rgb_vid_in_p):
        return
    if is_file_here(rgb_4d_out_p):
        rgb_4d_out_p = None
    if is_file_here(flow_4d_out_p):
        flow_4d_out_p = None
    if rgb_4d_out_p is None and flow_4d_out_p is None:
        return
    # Saves files to disk in format (time, height, width, channel) as numpy array
    flow_type = 1 # TVL1 optical flow
    #flow_type = None # will not process optical flow
    op = OpticalFlow(rgb_vid_in_p=rgb_vid_in_p, rgb_4d_out_p=rgb_4d_out_p,
            flow_4d_out_p=flow_4d_out_p, clip_flow_bound=20, flow_type=flow_type) # TVL1 optical flow
    op.process()


if __name__ == "__main__":
    main(sys.argv)
