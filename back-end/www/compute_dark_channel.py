import os
import sys
from util import *
import numpy as np
from multiprocessing import Pool


# Load processed rgb video frames and compute the dark channel
# Save the dark channel as the 4th channel with rgb together
# For dark channel, see paper "Single Image Haze Removal Using Dark Channel Prior"
# https://ieeexplore.ieee.org/abstract/document/5567108
# The input and output file format are numpy.array
def main(argv):
    rgb_dir = "../data/rgb/"
    rgbd_dir = "../data/rgbd/" # rgb + dark channel
    check_and_create_dir(rgb_dir)
    check_and_create_dir(rgbd_dir)
    file_names = get_all_file_names_in_folder(rgb_dir)
    p = Pool() # use all available CPUs
    p.map(compute_dark_channel, file_names)
    print("Done compute_dark_channel.py")


def compute_dark_channel(file_path):
    print("Process", file_path)
    rgb_dir = "../data/rgb/"
    rgbd_dir = "../data/rgbd/" # rgb + dark channel
    rgb = np.load(rgb_dir + file_path)
    s = rgb.shape
    d = np.empty((s[0],s[1],s[2]), dtype=rgb.dtype) # dark channel
    for i in range(s[0]):
        d[i,:,:] = get_dark_channel(rgb[i,...])
    rgbd = np.concatenate((rgb, np.expand_dims(d, axis=3)), axis=3)
    np.save(rgbd_dir + file_path, rgbd)


# Thid function is modified from the following link
# https://github.com/joyeecheung/dark-channel-prior-dehazing
def get_dark_channel(I, w=15):
    """Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, 255]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, 255]).
    """
    M, N, _ = I.shape
    padded = np.pad(I, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N), dtype=I.dtype)
    for i, j in np.ndindex(darkch.shape):
        # This is from equation 5 in the above mentioned dark channel paper
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])
    return darkch


if __name__ == "__main__":
    main(sys.argv)
