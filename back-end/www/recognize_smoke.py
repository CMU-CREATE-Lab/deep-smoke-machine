import os
import sys
from util import *
import numpy as np
from optical_flow.optical_flow import OpticalFlow
import urllib.request
from i3d_learner import I3dLearner
from grad_cam_viz import GradCam


# This is the production code for smoke recognition
def main(argv):
    vid_p = "../data/production/video/"
    rgb_p = "../data/production/rgb/"
    check_and_create_dir(rgb_p)
    check_and_create_dir(vid_p)

    url = "https://thumbnails-v2.createlab.org/thumbnail?root=http://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2019-02-03.timemachine/&boundsLTRB=5329,953,5831,1455&width=180&height=180&startFrame=7748&format=mp4&fps=12&tileFormat=mp4&nframes=50"
    file_name = "temp"
    """
    if is_file_here(rgb_p + file_name + ".npy"):
        print("File exists...skip downloading...")
    else:
        download_and_save(url, vid_p, rgb_p, file_name)
    """
    download_and_save(url, vid_p, rgb_p, file_name)
    recognize_smoke(rgb_p, file_name)


# Given a video url from the thumbnail server
# This function will download the video
# And also convert the video to numpy.array
# Then save the video and the numpy.array to a local file
def download_and_save(url, vid_p, rgb_p, file_name):
    rgb_vid_in_p = vid_p + file_name + ".mp4"
    rgb_4d_out_p = rgb_p + file_name + ".npy"
    try:
        print("Downloading video:", url)
        urllib.request.urlretrieve(url, rgb_vid_in_p)
    except:
        print("Error downloading:", url)
        return
    op = OpticalFlow(rgb_vid_in_p=rgb_vid_in_p, rgb_4d_out_p=rgb_4d_out_p, flow_type=None)
    op.process()


# The core function for smoke recognition
# Input:
# - file path that points to the numpy.array file
# - file name of the numpy.array file
# - the video numpy.array has shape (time, height, width, channel) = (n, 180, 180, 3)
# First output:
# - probabilities of having smoke, with shape (time, num_of_classes) = (n, 2)
# Second output:
# - the number of smoke pixels (using Grad-CAM), with shape (time, num_of_smoke_pixels)
def recognize_smoke(rgb_p, file_name):
    # Prepare model
    mode = "rgb"
    learner = I3dLearner(mode=mode, use_cuda=False)
    p_model = "../data/saved_i3d/paper_result/full-augm-rgb/55563e4-i3d-rgb-s3/model/573.pt"
    model = learner.set_model(0, 1, mode, p_model, False)
    transform = learner.get_transform(mode, image_size=224)

    # Prepare data
    v = np.load(rgb_p + file_name + ".npy")
    v = transform(v)
    v = torch.unsqueeze(v, 0)
    pred = learner.make_pred(model, v).squeeze().transpose(0, 1).detach().numpy()
    print(pred.shape)

    # Grad Cam
    grad_cam = GradCam(model)
    target_class = 1 # has smoke
    cam = grad_cam.generate_cam(v, target_class)
    print(cam.shape)


if __name__ == "__main__":
    main(sys.argv)
