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

from viz_functional import preprocess_image, save_class_activation_images
from torchvision import models
from i3d_learner import I3dLearner
from smoke_video_dataset import SmokeVideoDataset
from scipy.ndimage import zoom


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        if self.target_layer is None: # i3d model
            x = self.model.extract_conv_output(x)
            x.register_hook(self.save_gradient)
            conv_output = x  # Save the convolution output on that layer
        else: # alexnet model
            for module_pos, module in self.model.features._modules.items():
                x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        if self.target_layer is None: # i3d model
            x = self.model.conv_output_to_model_output(x)
        else: # alexnet model
            x = x.view(x.size(0), -1)  # Flatten
            x = self.model.classifier(x) # Forward pass on the classifier
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.extractor = CamExtractor(self.model, target_layer) # Define extractor

    def generate_cam(self, input_tensor, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_tensor)
        if target_class is None:
            target_class = 0
        if self.target_layer is None: # i3d model
            one_hot_output = torch.zeros_like(model_output)
            one_hot_output[0][target_class] = 1
            self.model.zero_grad()
        else: # alexnet model
            # Target for backprop
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
            one_hot_output[0][target_class] = 1
            # Zero grads
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        if self.target_layer is None: # i3d model
            weights = np.mean(guided_gradients, axis=(1, 2, 3))
        else: # alexnet model
            weights = np.mean(guided_gradients, axis=(1, 2)) # Take averages for each gradient
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
        if self.target_layer is None: # i3d model
            cam = zoom(cam, (i_sp[2]/c_sp[0], i_sp[3]/c_sp[1], i_sp[4]/c_sp[2]), order=1) / 255
        else: # alexnet model
            cam = np.uint8(Image.fromarray(cam).resize((i_sp[2], i_sp[3]), Image.ANTIALIAS)) / 255
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

    span = 4 # downample the time dimension
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


def main(argv):
    use_i3d = True
    if use_i3d: # i3d model
        mode = "rgb"
        learner = I3dLearner(mode=mode, use_cuda=False)
        metadata_path = "../data/split/metadata_train_split_0_by_camera.json"
        root_dir = "../data/rgb/"
        p_model = "../data/saved_i3d/146f769-i3d-rgb-s0/model/3042.pt"
        pretrained_model = learner.set_model(0, 1, mode, p_model, False)
        transform = learner.get_transform("rgb", image_size=224)
        dataset = SmokeVideoDataset(metadata_path=metadata_path, root_dir=root_dir, transform=transform)
        data = dataset[0]
        prep_input = torch.unsqueeze(data["frames"], 0)
        target_class = 1
        grad_cam = GradCam(pretrained_model)
        cam = grad_cam.generate_cam(prep_input, target_class)
        file_name_to_export = data["file_name"]
        save_class_activation_videos(prep_input, cam, file_name_to_export)
    else: # alexnet
        original_image = Image.open("../data/snake.jpg").convert('RGB')
        prep_input = preprocess_image(original_image)
        target_class = 56
        file_name_to_export = "snake"
        pretrained_model = models.alexnet(pretrained=True)
        grad_cam = GradCam(pretrained_model, target_layer=11)
        cam = grad_cam.generate_cam(prep_input, target_class)
        save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')


if __name__ == "__main__":
    main(sys.argv)
