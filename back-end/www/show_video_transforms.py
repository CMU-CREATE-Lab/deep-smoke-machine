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
import numpy as np
from base_learner import BaseLearner


class TestLearner(BaseLearner):
    def fit(self):
        pass

    def predict(self):
        pass

def main(argv):
    if len(argv) < 2:
        print("Usage: python show_video_transforms.py [video_file_path]")
        return

    # Read frames
    op = OpticalFlow(rgb_vid_in_p=argv[1])
    rgb_4d = op.vid_to_frames().astype(np.uint8) # ColorJitter need uint8
    tl = TestLearner()
    T = tl.get_transform("rgb", phase="train")
    rgb_4d = T(rgb_4d).numpy().transpose(1, 2, 3, 0)
    print(np.amin(rgb_4d), np.amax(rgb_4d))
    print(rgb_4d.shape)
    rgb_4d = cv2.normalize(rgb_4d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    op.frames_to_vid(rgb_4d, "../data/transformed.mp4")


if __name__ == "__main__":
    main(sys.argv)
