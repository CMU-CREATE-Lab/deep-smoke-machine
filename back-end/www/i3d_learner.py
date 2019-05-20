os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # use the order in the nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # specify which GPU(s) to be used
from base_learner import BaseLearner
from model.pytorch_i3d import InceptionI3d
from torch.utils.data import DataLoader
from smoke_video_dataset import SmokeVideoDataset
from model.pytorch_i3d import InceptionI3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

# Two-Stream Inflated 3D ConvNet learner
# https://arxiv.org/abs/1705.07750
class I3dLearner(BaseLearner):
    def __init__(self,
            batch_size=32, # size for each batch
            num_epochs=5, # number of epochs
            init_lr=0.001, # initial learning rate
            weight_decay=0.0000001, # L2 regularization
            momentum=0.9, # SGD parameters
            milestones=[2, 4], # MultiStepLR parameters
            gamma=0.1, # MultiStepLR parameters
            num_workers=4)
        super().__init__()
        self.create_logger(log_path="I3dLearner.log")
        self.log("Use Two-Stream Inflated 3D ConvNet learner")

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.num_workers = num_workers

    def fit(self,
            mode="rgb",
            p_metadata_train="../data/metadata_train.json",
            p_metadata_validation="../data/metadata_validation.json",
            p_vid="../data/videos/",
            p_pretrain="../data/pretrained_models/"):

        # Setup the model and load pre-trained weights
        if mode == "rgb":
            i3d = InceptionI3d(400, in_channels=3)
            i3d.load_state_dict(torch.load(p_pretrain+"i3d_rgb_imagenet_kinetics.pt"))
        elif mode == "flow":
            i3d = InceptionI3d(400, in_channels=2)
            i3d.load_state_dict(torch.load(p_pretrain+"i3d_flow_imagenet_kinetics.pt"))
        else:
            return None
        i3d.replace_logits(157)

        # Use GPU or not
        if self.use_cuda:
            i3d.cuda()

        # Load datasets
        print("Create train dataset...")
        ds_t = SmokeVideoDataset(metadata_path=p_metadata_train, root_dir=p_vid, mode=mode)
        print("Create train dataloader")
        dl_t = DataLoader(ds_t, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        print("Create validation dataset...")
        ds_v = SmokeVideoDataset(metadata_path=p_metadata_validation, root_dir=p_vid, mode=mode)
        print("Create validation dataloader")
        dl_v = DataLoader(ds_v, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        # Set optimizer
        optimizer = optim.SGD(i3d.parameters(), lr=self.init_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

    def predict(self, X):
        pass
