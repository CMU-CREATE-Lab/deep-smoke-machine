from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import os
import logging
import absl.logging
import logging.handlers
from util import check_and_create_dir
from collections import OrderedDict
from torchvision.transforms import Compose
from video_transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomPerspective, RandomErasing, Resize, Normalize, ToTensor


class RequestFormatter(logging.Formatter):
    def format(self, record):
        return super().format(record)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        """
        Reshapes the input according to the shape saved in the view data structure.
        """
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.reshape(shape)
        return out


"""
Base PyTorch learners
Usage:
    from base_pytorch_learner import BasePyTorchLearner

    class Learner(BasePyTorchLearner):
        def __init__(self):
            super().__init__()
            self.create_logger(log_path="../log/Learner.log")

    def fit(self, Xt, Yt, Xv=None, Yv=None):
        pass

    def predict(self, X):
        pass
"""
class BaseLearner(ABC):
    def __init__(self, use_cuda=None):
        self.logger = None
        if use_cuda is None:
            if torch.cuda.is_available:
                self.use_cuda = True
            else:
                self.use_cuda = False
        else:
            if use_cuda is True and torch.cuda.is_available:
                self.use_cuda = True
            else:
                self.use_cuda = False

    # Train the model
    # Output: None
    @abstractmethod
    def fit(self):
        pass

    # Test the model
    # Output: None
    @abstractmethod
    def test(self):
        pass

    # Save model
    def save(self, model, out_path):
        if model is not None and out_path is not None:
            self.log("Save model weights to " + out_path)
            try:
                state_dict = model.module.state_dict() # nn.DataParallel model
            except AttributeError:
                state_dict = model.state_dict() # single GPU model
            check_and_create_dir(out_path)
            torch.save(state_dict, out_path)

    # Load model
    def load(self, model, in_path, ignore_fc=False, fill_dim=False):
        if model is not None and in_path is not None:
            self.log("Load model weights from " + in_path)
            sd_loaded = torch.load(in_path)
            if "state_dict" in sd_loaded:
                sd_loaded = sd_loaded["state_dict"]
            sd_model = model.state_dict()
            replace_dict = []
            for k, v in sd_loaded.items():
                if k not in sd_model and k.replace(".net", "") in sd_model:
                    print("Load after remove .net: ", k)
                    replace_dict.append((k, k.replace(".net", "")))
            for k, v in sd_model.items():
                if k not in sd_loaded and k.replace(".net", "") in sd_loaded:
                    print("Load after adding .net: ", k)
                    replace_dict.append((k.replace(".net", ""), k))
            for k, k_new in replace_dict:
                sd_loaded[k_new] = sd_loaded.pop(k)
            keys1 = set(list(sd_loaded.keys()))
            keys2 = set(list(sd_model.keys()))
            set_diff = (keys1 - keys2) | (keys2 - keys1)
            #print('#### Notice: keys that failed to load: {}'.format(set_diff))
            if ignore_fc:
                print("Ignore fully connected layer weights")
                sd_loaded = {k: v for k, v in sd_model.items() if "fc" not in k}
            if fill_dim:
                # Note that this only works for the Inception-v1 I3D model
                print("Auto-fill the mismatched dimension for the i3d model...")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if param.data.size() != sd_loaded[name].size():
                            print("\t Found dimension mismatch for:", name)
                            ds = param.data.size()
                            ls = sd_loaded[name].size()
                            print("\t\t Desired data size:", param.data.size())
                            print("\t\t Loaded data size:", sd_loaded[name].size())
                            for i in range(len(ds)):
                                diff = ds[i] - ls[i]
                                if diff > 0:
                                    print("\t\t\t Desired dimension %d is larger than the loaded dimension" % i)
                                    m = sd_loaded[name].mean(i).unsqueeze(i)
                                    print("\t\t\t Compute the missing dimension to have size:", m.size())
                                    sd_loaded[name] = torch.cat([sd_loaded[name], m], i)
                                    print("\t\t\t Loaded data are filled to have size:", sd_loaded[name].size())
            sd_model.update(sd_loaded)
            try:
                model.load_state_dict(sd_model)
            except:
                self.log("Weights were from nn.DataParallel or DistributedDataParallel...")
                self.log("Remove 'module.' prefix from state_dict keys...")
                new_state_dict = OrderedDict()
                for k, v in sd_model.items():
                    new_state_dict[k.replace("module.", "")] = v
                model.load_state_dict(new_state_dict)

    # Log information
    def log(self, msg, lv="i"):
        print(msg)
        if self.logger is not None:
            if lv == "i":
                self.logger.info(msg)
            elif lv == "w":
                self.logger.warning(msg)
            elif lv == "e":
                self.logger.error(msg)

    # Data augmentation pipeline
    def get_transform(self, mode, phase=None, image_size=224):
        if mode == "rgb": # two channels (r, g, and b)
            mean = (127.5, 127.5, 127.5)
            std = (127.5, 127.5, 127.5)
        elif mode == "flow": # two channels (x and y)
            mean = (127.5, 127.5)
            std = (127.5, 127.5)
        else:
            return None
        nm = Normalize(mean=mean, std=std) # same as (img/255)*2-1
        tt = ToTensor()
        if phase == "train":
            # Deals with small camera shifts, zoom changes, and rotations due to wind or maintenance
            rrc = RandomResizedCrop(image_size, scale=(0.9, 1), ratio=(3./4., 4./3.))
            rp = RandomPerspective(anglex=3, angley=3, anglez=3, shear=3)
            # Improve generalization
            rhf = RandomHorizontalFlip(p=0.5)
            # Deal with dirts, ants, or spiders on the camera lense
            re = RandomErasing(p=0.5, scale=(0.003, 0.01), ratio=(0.3, 3.3), value=0)
            if mode == "rgb":
                # Color jitter deals with different lighting and weather conditions
                cj = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1, 0.1), gamma=0.3)
                return Compose([cj, rrc, rp, rhf, tt, nm, re, re])
            elif mode == "flow":
                return Compose([rrc, rp, rhf, tt, nm, re, re])
        else:
            return Compose([Resize(image_size), tt, nm])

    # Create a logger
    def create_logger(self, log_path=None):
        if log_path is None:
            return None
        check_and_create_dir(log_path)
        handler = logging.handlers.RotatingFileHandler(log_path, mode="a", maxBytes=100000000, backupCount=200)
        logging.root.removeHandler(absl.logging._absl_handler) # this removes duplicated logging
        absl.logging._warn_preinit_stderr = False # this removes duplicated logging
        formatter = RequestFormatter("[%(asctime)s] %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger = logging.getLogger(log_path)
        logger.setLevel(logging.INFO)
        for hdlr in logger.handlers[:]:
            logger.removeHandler(hdlr) # remove old handlers
        logger.addHandler(handler)
        self.logger = logger
