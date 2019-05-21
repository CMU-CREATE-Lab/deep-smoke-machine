import os
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
            batch_size=8, # size for each batch
            num_epochs=5, # total number of epochs for training
            num_steps_per_update=1, # gradient accumulation (for large batch size that does not fit into memory)
            init_lr=0.001, # initial learning rate
            weight_decay=0.0000001, # L2 regularization
            momentum=0.9, # SGD parameters
            milestones=[2, 4], # MultiStepLR parameters
            gamma=0.1, # MultiStepLR parameters
            num_of_action_classes=2, # currently we only have two classes (0 and 1, which means no and yes)
            num_workers=4):
        super().__init__()
        self.create_logger(log_path="I3dLearner.log")
        self.log("Use Two-Stream Inflated 3D ConvNet learner")

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_steps_per_update = num_steps_per_update
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.milestones = milestones
        self.gamma = gamma
        self.num_of_action_classes = num_of_action_classes
        self.num_workers = num_workers

    def fit(self,
            mode="rgb",
            p_metadata_train="../data/metadata_train.json",
            p_metadata_validation="../data/metadata_validation.json",
            p_vid="../data/videos/",
            p_pretrain_rgb="../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt",
            p_pretrain_flow="../data/pretrained_models/i3d_flow_imagenet_kinetics.pt"):

        # Setup the model and load pre-trained weights
        if mode == "rgb":
            i3d = InceptionI3d(400, in_channels=3)
            i3d.load_state_dict(torch.load(p_pretrain_rgb))
        elif mode == "flow":
            i3d = InceptionI3d(400, in_channels=2)
            i3d.load_state_dict(torch.load(p_pretrain_flow))
        else:
            return None

        # Set the number of output classes in the model
        i3d.replace_logits(self.num_of_action_classes)

        # Use GPU or not
        if self.use_cuda:
            i3d.cuda()

        # Load datasets
        metadata_path = {"train": p_metadata_train, "validation": p_metadata_validation}
        dataset = {}
        dataloader = {}
        for phase in ["train", "validation"]:
            print("Create dataset for", phase)
            dataset[phase] = SmokeVideoDataset(metadata_path=metadata_path[phase], root_dir=p_vid, mode=mode)
            print("Create dataloader for", phase)
            dataloader[phase] = DataLoader(dataset[phase], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        # Set optimizer
        optimizer = optim.SGD(i3d.parameters(), lr=self.init_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

        # Train
        train_steps = 0
        for epoch in range(self.num_epochs):
            self.log("-"*40)
            self.log("Epoch %r/%r" % (epoch, self.num_epochs))
            # Each epoch has a training and validation phase
            for phase in ["train", "validation"]:
                if phase == "train":
                    i3d.train(True) # set model to training mode
                else:
                    i3d.train(False) # set model to evaluate mode
                tot_loss = 0.0 # total loss
                tot_loc_loss = 0.0 # total localization loss
                tot_cls_loss = 0.0 # total classification loss
                accum_count = 0 # counter for accumulating gradients
                optimizer.zero_grad()
                # Iterate over data
                for d in dataloader[phase]:
                    accum_count += 1
                    # Get inputs
                    frames = d["frames"]
                    if self.use_cuda:
                        frames = frames.cuda()
                    frames = Variable(frames)
                    label = d["label"]
                    if self.use_cuda:
                        label = label.cuda()
                    label = Variable(label)
                    # Upsample to frame length (because we want prediction for each frame)
                    per_frame_logits = i3d(frames)
                    per_frame_logits = F.interpolate(per_frame_logits, frames.size(2), mode="linear", align_corners=True)
                    # Compute localization loss
                    print(per_frame_logits.size())
                    print(label.size())
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, label)
                    tot_loc_loss += loc_loss.data[0]
                    return
                    # Backprop
                    loss = loc_loss / self.num_steps_per_update
                    tot_loss += loss.data
                    loss.backward()
                    # Accumulate gradients during training
                    if (accum_count == self.num_steps_per_update) and phase == "train":
                        train_steps += 1
                        accum_count = 0
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        if train_steps % 10 == 0:
                            print("%s loss: %.4f" % (phase, tot_loss/10)) # print training loss
                            tot_loss = tot_loc_loss = tot_cls_loss = 0.
                if phase == "validation":
                    print("%s loss: %.4f" % (phase, (tot_loss*self.num_steps_per_update)/accum_count)) # print validation loss

        self.log("Done fit")

    def predict(self, X):
        pass
