import os
import sys
from util import *
import numpy as np


# This is the production code for smoke recognition
def main(argv):
    url = "https://thumbnails-v2.createlab.org/thumbnail?root=http://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2019-02-03.timemachine/&boundsLTRB=5329,953,5831,1455&width=180&height=180&startFrame=7748&format=mp4&fps=12&tileFormat=mp4&nframes=360"
    save_path = "../data/production/"
    check_and_create_dir(save_path)
    download_and_save(url, save_path)


# Given a video url from the thumbnail server
# This function will download the video
# And also convert the video to numpy.array
# Then save the numpy.array to a local file
def download_and_save(url, save_path):
    pass


# The core function for smoke recognition
# Input:
# - a video numpy.array with shape (time, height, width, channel) = (n, 180, 180, 3)
# First output:
# - probabilities of having smoke, with shape (time, num_of_classes) = (n, 2)
# Second output:
# - the number of smoke pixels (using Grad-CAM), with shape (time, num_of_smoke_pixels)
def recognize_smoke(V):
    pass


if __name__ == "__main__":
    main(sys.argv)
