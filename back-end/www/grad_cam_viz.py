"""
Modified from https://github.com/utkuozbulak/pytorch-cnn-visualizations
"""
from PIL import Image
import numpy as np
import torch

from viz_functional import preprocess_image, save_class_activation_images, save_class_activation_videos
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


if __name__ == '__main__':
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
