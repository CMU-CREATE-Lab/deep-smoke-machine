#TODO: add the early stopping logic
# if the validation error increases for n consecutive times, stop training

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # use the order in the nvidia-smi command
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # specify which GPU(s) to be used
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # specify which GPU(s) to be used
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
from util import check_and_create_dir
import uuid
from sklearn.metrics import classification_report
import numpy as np

# Two-Stream Inflated 3D ConvNet learner
# https://arxiv.org/abs/1705.07750
class I3dLearner(BaseLearner):
    def __init__(self,
            batch_size=16, # size for each batch (8 for each GTX 1080Ti)
            max_steps=64e3, # total number of steps for training
            num_steps_per_update=4, # gradient accumulation (for large batch size that does not fit into memory)
            init_lr=0.001, # initial learning rate
            weight_decay=0.0000001, # L2 regularization
            momentum=0.9, # SGD parameters
            milestones=[300, 1000], # MultiStepLR parameters (steps for decreasing the learning rate)
            gamma=0.1, # MultiStepLR parameters
            num_of_action_classes=2, # currently we only have two classes (0 and 1, which means no and yes)
            save_model_path="saved_i3d/", # path for saving the models
            num_steps_per_check=10, # the number of steps to save a model and log information
            parallel=True, # use nn.DataParallel or not
            num_workers=2):
        super().__init__()
        self.create_logger(log_path="I3dLearner.log")
        self.log("Use Two-Stream Inflated 3D ConvNet learner")

        self.batch_size = batch_size
        self.max_steps = max_steps
        self.num_steps_per_update = num_steps_per_update
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.milestones = milestones
        self.gamma = gamma
        self.num_of_action_classes = num_of_action_classes
        check_and_create_dir(save_model_path)
        self.save_model_path = save_model_path
        self.num_steps_per_check = num_steps_per_check
        self.parallel = parallel
        self.num_workers = num_workers

    def fit(self,
            mode="rgb",
            p_metadata_train="../data/metadata_train.json",
            p_metadata_validation="../data/metadata_validation.json",
            p_vid="../data/videos/",
            p_pretrain_rgb="../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt",
            p_pretrain_flow="../data/pretrained_models/i3d_flow_imagenet_kinetics.pt"):

        self.log("="*60)
        self.log("="*60)
        self.log("Start training...")

        # Setup the model and load pre-trained weights
        if mode == "rgb":
            i3d = InceptionI3d(400, in_channels=3)
            self.load(i3d, p_pretrain_rgb)
        elif mode == "flow":
            i3d = InceptionI3d(400, in_channels=2)
            self.load(i3d, p_pretrain_flow)
        else:
            return None

        # Set the number of output classes in the model
        i3d.replace_logits(self.num_of_action_classes)

        # Use GPU or not
        if self.use_cuda:
            i3d.cuda()
            if self.parallel and torch.cuda.device_count() > 1:
                self.log("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
                i3d = nn.DataParallel(i3d)

        # Load datasets
        metadata_path = {"train": p_metadata_train, "validation": p_metadata_validation}
        dataset = {}
        dataloader = {}
        for phase in ["train", "validation"]:
            self.log("Create dataset for", phase)
            dataset[phase] = SmokeVideoDataset(metadata_path=metadata_path[phase], root_dir=p_vid, mode=mode)
            self.log("Create dataloader for", phase)
            dataloader[phase] = DataLoader(dataset[phase], batch_size=self.batch_size,
                    shuffle=True, num_workers=self.num_workers, pin_memory=True)

        # Set optimizer
        optimizer = optim.SGD(i3d.parameters(), lr=self.init_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_sche= optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

        # Set logging format
        log_fm = "%s step: %d lr: %r loc_loss: %.4f cls_loss: %.4f loss: %.4f"

        # Train and validate
        steps = 0
        epochs = 0
        nspu = self.num_steps_per_update
        nspc = self.num_steps_per_check
        model_id = str(uuid.uuid4())[0:7]
        accum = {} # counter for accumulating gradients
        tot_loss = {} # total loss
        tot_loc_loss = {} # total localization loss
        tot_cls_loss = {} # total classification loss
        pred_labels = {} # predicted labels
        true_labels = {} # true labels
        for phase in ["train", "validation"]:
            accum[phase] = 0
            tot_loss[phase] = 0.0
            tot_loc_loss[phase] = 0.0
            tot_cls_loss[phase] = 0.0
            pred_labels[phase] = []
            true_labels[phase] = []
        while steps < self.max_steps:
            self.log("-"*40)
            # Each epoch has a training and validation phase
            for phase in ["train", "validation"]:
                self.log("phase " + phase)
                if phase == "train":
                    epochs += 1
                    self.log("epochs: %d steps: %d/%d" % (epochs, steps, self.max_steps))
                    i3d.train(True) # set model to training mode
                else:
                    i3d.train(False) # set model to evaluate mode
                optimizer.zero_grad()
                # Iterate over data
                for d in dataloader[phase]:
                    accum[phase] += 1
                    # Get inputs
                    frames = d["frames"]
                    if self.use_cuda:
                        frames = frames.cuda()
                    frames = Variable(frames)
                    labels = d["labels"]
                    # Save true labels
                    true_labels[phase] += np.argmax(labels.numpy().max(axis=2), axis=1).tolist()
                    # Move to GPU
                    if self.use_cuda:
                        labels = labels.cuda()
                    labels = Variable(labels)
                    # Upsample prediction to frame length (because we want prediction for each frame)
                    pred = i3d(frames)
                    pred = F.interpolate(pred, frames.size(2), mode="linear", align_corners=True)
                    # Save predicted labels
                    pred_labels[phase] += np.argmax(pred.cpu().detach().numpy().max(axis=2), axis=1).tolist()
                    # Compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(pred, labels)
                    tot_loc_loss[phase] += loc_loss.data
                    # Compute classification loss (with max-pooling along time, batch x channel x time)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(pred, dim=2)[0], torch.max(labels, dim=2)[0])
                    tot_cls_loss[phase] += cls_loss.data
                    # Backprop
                    loss = (0.5*loc_loss + 0.5*cls_loss) / nspu
                    tot_loss[phase] += loss.data
                    loss.backward()
                    # Accumulate gradients during training
                    if (accum[phase] == nspu) and phase == "train":
                        steps += 1
                        accum[phase] = 0
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_sche.step()
                        if steps % nspc == 0:
                            lr = lr_sche.get_lr()[0]
                            tll = tot_loc_loss[phase]/(nspc*nspu)
                            tcl = tot_cls_loss[phase]/(nspc*nspu)
                            tl = tot_loss[phase]/nspc
                            self.log(log_fm % (phase, steps, lr, tll, tcl, tl))
                            tot_loss[phase] = tot_loc_loss[phase] = tot_cls_loss[phase] = 0.0
                if phase == "validation":
                    lr = lr_sche.get_lr()[0]
                    tll = tot_loc_loss[phase]/accum[phase]
                    tcl = tot_cls_loss[phase]/accum[phase]
                    tl = (tot_loss[phase]*nspu)/accum[phase]
                    self.log(log_fm % (phase, steps, lr, tll, tcl, tl))
                    tot_loss[phase] = tot_loc_loss[phase] = tot_cls_loss[phase] = 0.0
                    accum[phase] = 0
                    model_p = self.save_model_path + model_id + "-" + str(steps) + ".pt"
                    self.save(i3d, model_p)
                    self.log("save model to " + model_p)
                    for phase in ["train", "validation"]:
                        self.log("Performance for " + phase)
                        self.log(classification_report(true_labels[phase], pred_labels[phase]))
                        pred_labels[phase] = []
                        true_labels[phase] = []

        self.log("Done fit")

    def predict(self, X):
        self.log("predict")
        pass
