"""
Modified from https://github.com/utkuozbulak/pytorch-cnn-visualizations
"""
from PIL import Image
import matplotlib.cm as mpl_color_map
import numpy as np
import torch
import sys
import os
import copy

from torchvision import models
from i3d_learner import I3dLearner
from scipy.ndimage import zoom
from util import *
import re


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at the last conv layer
        """
        x = self.model.extract_conv_output(x)
        x.register_hook(self.save_gradient)
        conv_output = x  # Save the convolution output on the last conv layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = self.model.conv_output_to_model_output(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model) # Define extractor

    def generate_cam(self, input_tensor, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_tensor)
        if target_class is None:
            target_class = 0
        # Target for backprop
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2, 3)) # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, ...]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        i_sp = input_tensor.shape
        c_sp = cam.shape
        # Scale the map up to the input tensor size
        cam = zoom(cam, (i_sp[2]/c_sp[0], i_sp[3]/c_sp[1], i_sp[4]/c_sp[2]), order=1) / 255
        return cam


def save_class_activation_videos(org_vid, activation_map, file_name, root_dir="../data/cam"):
    """
        Saves cam activation map and activation map on the original video

    Args:
        org_vid (numpy.ndarray): Original video with dimension batch*channel*time*height*width
        activation_map (umpy.ndarray): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    span = 9 # downample the time dimension
    org_vid = org_vid[:, :, ::span, :, :]
    activation_map = activation_map[::span, :, :]

    color_map = mpl_color_map.get_cmap("jet") # get color map
    no_trans_heatmap = color_map(activation_map)

    activation_map = np.expand_dims(activation_map, axis=3)
    activation_map = convert_3d_to_2d(activation_map, constant_values=0)
    activation_map = activation_map[:, :, 0]

    heatmap = copy.deepcopy(no_trans_heatmap)
    heatmap = convert_3d_to_2d(heatmap, constant_values=1)
    heatmap[:, :, 3] = np.minimum(np.maximum(activation_map*2 - 1, 0), 1) # change alpha to show the original image
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))

    no_trans_heatmap = convert_3d_to_2d(no_trans_heatmap, constant_values=0)
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    org_vid = np.transpose(org_vid[0, ...], (1, 2, 3, 0)) # do not use the batch dimension
    org_vid = convert_3d_to_2d(org_vid, constant_values=1)
    org_vid = Image.fromarray(((org_vid+1)*127.5).astype(np.uint8))

    heatmap_on_image = Image.new("RGBA", org_vid.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_vid.convert("RGBA"))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)

    stacked = Image.new("RGB", (org_vid.size[0], org_vid.size[1]*2 + 20), (255, 255, 255))
    stacked.paste(org_vid, (0, 0))
    stacked.paste(heatmap_on_image, (0, org_vid.size[1] + 20))

    #no_trans_heatmap.save(os.path.join(root_dir, file_name+"-cam-heatmap.png"))
    #org_vid.save(os.path.join(root_dir, file_name+"-video.png"))
    #heatmap_on_image.save(os.path.join(root_dir, file_name+"-cam-on-video.png"))
    stacked.save(os.path.join(root_dir, file_name+"-cam-stacked.png"))


# Flatten a numpy.ndarray with dimension (time*height*width*channel)
def convert_3d_to_2d(frames, constant_values=0):
    frames = np.transpose(frames, (1, 0, 2, 3))
    pad_w = 20
    npad = ((0, 0), (0, 0), (0, pad_w), (0, 0))
    frames = np.pad(frames, pad_width=npad, mode="constant", constant_values=constant_values) # add padding
    sp = frames.shape
    frames = np.reshape(frames, (sp[0], sp[1]*sp[2], sp[3])) # 3d to 2d
    frames = frames[:, :-20, :] # remove padding for the last frame
    return frames


def grad_cam(p_model):
    mode = "rgb"
    p_frame = "../data/rgb/"
    n = 128 # number of videos per set (TP, TN, FP, FN)

    # Check
    if p_model is None or not is_file_here(p_model):
        self.log("Need to provide a valid model path")
        return

    # Set path
    match = re.search(r'\b/[0-9a-fA-F]{7}-i3d-(rgb|flow)[^/]*/\b', p_model)
    model_id = match.group()[1:-1]
    if model_id is None:
        self.log("Cannot find a valid model id from the model path.")
        return
    p_root = p_model[:match.start()] + "/" + model_id + "/"
    p_metadata_test = p_root + "metadata/metadata_test.json" # metadata path (test)
    save_viz_path = p_root + "viz/" # path to save visualizations

    # Set model
    learner = I3dLearner(mode=mode, use_cuda=False)
    pretrained_model = learner.set_model(0, 1, mode, p_model, False)

    # Select samples and generate class activation maps
    transform = learner.get_transform(mode, image_size=224)
    cm = load_json(save_viz_path + "0/confusion_matrix_of_samples.json")
    for u in cm:
        for v in cm[u]:
            n_uv = np.minimum(len(cm[u][v]), n)
            samples = np.random.choice(cm[u][v], n_uv)
            p_cam = p_root + ("cam/true_%s_prediction_%s/" % (u, v))
            check_and_create_dir(p_cam)
            print("Prepare folder %s" % (p_cam))
            # Generate cam
            for file_name in samples:
                print("Process file %s" % (file_name))
                prep_input = np.load(p_frame + file_name + ".npy")
                prep_input = transform(prep_input)
                prep_input = torch.unsqueeze(prep_input, 0)
                target_class = 1 # has smoke
                grad_cam = GradCam(pretrained_model)
                cam = grad_cam.generate_cam(prep_input, target_class)
                save_class_activation_videos(prep_input, cam, file_name, root_dir=p_cam)
    print('Grad cam completed')


def main(argv):
    if len(argv) < 3:
        print("Usage: python grad_cam_viz.py [method] [model_path]")
        return
    method = argv[1]
    model_path = argv[2]
    if method is None or model_path is None:
        print("Usage: python grad_cam_viz.py [method] [model_path]")
        return
    if method == "i3d-rgb":
        grad_cam(model_path)
    else:
        print("Method not allowed")


if __name__ == "__main__":
    main(sys.argv)
