import sys
from util import *
import cv2 as cv
import numpy as np
from optical_flow import OpticalFlow

# Process videos into rgb frame files and optical flow files
# The file format is numpy.array
def main(argv):
    op = OpticalFlow()

    rgb_dir = "../data/rgb/"
    flow_dir = "../data/flow/"
    metadata_path = "../data/metadata.json"
    video_dir = "../data/videos/"

    # Check for saving directories and create if they don't exist
    check_and_create_dir(rgb_dir)
    check_and_create_dir(flow_dir)

    metadata = load_json(metadata_path)
    for video_data in metadata:
        file_name = video_data["file_name"]
        if is_file_here(rgb_dir + file_name + ".npy") and is_file_here(flow_dir + file_name + ".npy"): continue
        video = str(video_dir + file_name + ".mp4")
        rgb_4d_out_p = str(rgb_dir + file_name)
        flow_4d_out_p = str(flow_dir + file_name)
        # Saves files to disk in format (time, height, width, channel) as numpy array
        op.step(rgb_vid_in_p=video, rgb_4d_out_p=rgb_4d_out_p, flow_4d_out_p=flow_4d_out_p)

if __name__ == "__main__":
    main(sys.argv)
