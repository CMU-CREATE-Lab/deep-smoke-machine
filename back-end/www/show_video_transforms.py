import sys
import matplotlib
matplotlib.use("TkAgg") # a fix for Mac OS X error
from optical_flow.optical_flow import OpticalFlow
from torchvision.transforms import Compose
from video_transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomPerspective, RandomErasing
import numpy as np


def main(argv):
    if len(argv) < 2:
        print("Usage: python test_video_transforms.py [video_file_path]")
        return

    # Read frames
    op = OpticalFlow(rgb_vid_in_p=argv[1])
    rgb_4d = op.vid_to_frames().astype(np.uint8) # ColorJitter need uint8

    # Color jitter deals with different lighting and weather conditions
    cj = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1, 0.1))

    # Deals with small camera shifts, zoom changes, and rotations due to wind or maintenance
    rrc = RandomResizedCrop(224, scale=(0.9, 1), ratio=(3./4., 4./3.))
    rp = RandomPerspective(anglex=3, angley=3, anglez=3, shear=3)

    # Improve generalization
    rhf = RandomHorizontalFlip(p=0.5)

    # Deal with dirts, ants, or spiders on the camera lense
    re = RandomErasing(p=0.5, scale=(0.001, 0.007), ratio=(0.3, 3.3), value="random")

    # Transform and save
    T = Compose([cj, rrc, rp, rhf, re, re, re])
    rgb_4d = T(rgb_4d)
    op.frames_to_vid(rgb_4d, "../data/transformed.mp4")


if __name__ == "__main__":
    main(sys.argv)
