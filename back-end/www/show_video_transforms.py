import sys
import matplotlib
matplotlib.use("TkAgg") # a fix for Mac OS X error
from optical_flow.optical_flow import OpticalFlow
from torchvision.transforms import Compose
from video_transforms import RandomCrop, RandomHorizontalFlip, ColorJitter
import numpy as np


def main(argv):
    if len(argv) < 2:
        print("Usage: python test_video_transforms.py [video_file_path]")
        return
    op = OpticalFlow(rgb_vid_in_p=argv[1])
    rgb_4d = op.vid_to_frames().astype(np.uint8) # ColorJitter need uint8
    rc = RandomCrop(180)
    rf = RandomHorizontalFlip(p=0.5)
    cj = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1, 0.1))
    T = Compose([rc, rf, cj])
    rgb_4d = T(rgb_4d)
    op.frames_to_vid(rgb_4d, "../data/transformed.mp4")


if __name__ == "__main__":
    main(sys.argv)
