import os
thread = "1"
os.environ["MKL_NUM_THREADS"] = thread
os.environ["NUMEXPR_NUM_THREADS"] = thread
os.environ["OMP_NUM_THREADS"] = thread
os.environ["VECLIB_MAXIMUM_THREADS"] = thread
os.environ["OPENBLAS_NUM_THREADS"] = thread
import cv2
cv2.setNumThreads(0)

import sys
import matplotlib
matplotlib.use("TkAgg") # a fix for Mac OS X error
from optical_flow.optical_flow import OpticalFlow
from torchvision.transforms import Compose
from video_transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomPerspective, RandomErasing, Normalize, Resize
import numpy as np


def main(argv):
    if len(argv) < 2:
        print("Usage: python test_video_transforms.py [video_file_path]")
        return

    # Read frames
    op = OpticalFlow(rgb_vid_in_p=argv[1])
    rgb_4d = op.vid_to_frames().astype(np.uint8) # ColorJitter need uint8

    # Color jitter deals with different lighting and weather conditions
    cj = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1, 0.1), gamma=0.3)

    # Deals with small camera shifts, zoom changes, and rotations due to wind or maintenance
    rrc = RandomResizedCrop(224, scale=(0.9, 1), ratio=(3./4., 4./3.))
    rp = RandomPerspective(anglex=3, angley=3, anglez=3, shear=3)

    # Improve generalization
    rhf = RandomHorizontalFlip(p=0.5)

    # Deal with dirts, ants, or spiders on the camera lense
    re = RandomErasing(p=0.5, scale=(0.002, 0.008), ratio=(0.3, 3.3), value="random")

    # Normalization
    nm = Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)) # same as (img/255)*2-1

    # Transform and save
    T = Compose([cj, rrc, rp, rhf, re, re, nm])
    #T = Compose([Resize(224)])
    rgb_4d = T(rgb_4d)
    print(rgb_4d.shape)
    rgb_4d = cv2.normalize(rgb_4d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    op.frames_to_vid(rgb_4d, "../data/transformed.mp4")


if __name__ == "__main__":
    main(sys.argv)
