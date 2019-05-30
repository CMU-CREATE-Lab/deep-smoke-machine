import sys
from util import *
import cv2 as cv
import numpy as np
from optical_flow import OpticalFlow

# Process videos into rgb frame files and optical flow files
# The file format is numpy.array
def main(argv):
    # Check
    if len(argv) > 1:
        if argv[1] != "confirm":
            print("Must confirm by running: python process_videos.py confirm")
            return
    else:
        print("Must confirm by running: python process_videos.py confirm")
        return

    # TODO: load video metadata from ../data/metadata.json

    # TODO: loop through the metadata to get the file_name
        # TODO: Skip this file if ../data/frames/[file_name].npy and ../data/flows/[file_name].npy both exist
        # TODO: for example: if is_file_here(file_path): continue
        # TODO: load videos from ../data/videos/[file_name].mp4
        # TODO: process them into rgb frames and optical flows
        # TODO: save rgb frames to ../data/frames/[file_name].npy
        # TODO: save optical flows to ../data/flows/[file_name].npy

if __name__ == "__main__":
    main(sys.argv)
