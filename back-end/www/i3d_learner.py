import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # use the order in the nvidia-smi command
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # specify which GPU(s) to be used
os.environ["CUDA_VISIBLE_DEVICES"]="0,2" # specify which GPU(s) to be used
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
import uuid
from sklearn.metrics import classification_report
import numpy as np
from torchvision import transforms
from video_transforms import *

# Two-Stream Inflated 3D ConvNet learner
# https://arxiv.org/abs/1705.07750
class I3dLearner(BaseLearner):
    def __init__(self,
            batch_size=16, # size for each batch (8 max for each GTX 1080Ti)
            max_steps=64e3, # total number of steps for training
            num_steps_per_update=4, # gradient accumulation (for large batch size that does not fit into memory)
            init_lr=0.001, # initial learning rate
            weight_decay=0.0000001, # L2 regularization
            momentum=0.9, # SGD parameters
            milestones=[2000, 4000], # MultiStepLR parameters (steps for decreasing the learning rate)
            gamma=0.1, # MultiStepLR parameters
            num_of_action_classes=2, # currently we only have two classes (0 and 1, which means no and yes)
            save_model_path="../data/saved_i3d/", # path for saving the models
            num_steps_per_check=10, # the number of steps to save a model and log information
            parallel=True, # use nn.DataParallel or not
            augment=True, # use data augmentation or not
            num_workers=4):
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
        self.save_model_path = save_model_path
        self.num_steps_per_check = num_steps_per_check
        self.parallel = parallel
        self.augment = augment
        self.num_workers = num_workers

    def set_model(self, mode, p_pretrain):
        # Setup the model based on mode
        if mode == "rgb":
            model = InceptionI3d(400, in_channels=3)
        elif mode == "flow":
            model = InceptionI3d(400, in_channels=2)
        else:
            return None

        # Load i3d pre-trained weights
        is_self_trained = False
        try:
            if p_pretrain is not None:
                self.load(model, p_pretrain)
        except:
            # Not pre-trained weights
            is_self_trained = True

        # Set the number of output classes in the model
        model.replace_logits(self.num_of_action_classes)

        # Load self-trained weights from fine-tuned models
        if p_pretrain is not None and is_self_trained:
            self.load(model, p_pretrain)

        # Use GPU or not
        if self.use_cuda:
            model.cuda()
            if self.parallel and torch.cuda.device_count() > 1:
                self.log("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
                model = nn.DataParallel(model)

        return model

    def set_dataloader(self, metadata_path, p_frame, mode, tf):
        dataloader = {}
        for phase in metadata_path:
            self.log("Create dataloader for " + phase)
            dataset = SmokeVideoDataset(metadata_path=metadata_path[phase], root_dir=p_frame, mode=mode, transform=tf[phase])
            dataloader[phase] = DataLoader(dataset, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.num_workers, pin_memory=True)

        return dataloader

    def labels_to_list(self, labels):
        # Convert labels obtained from the dataloader to a list of action classes
        return np.argmax(labels.numpy().max(axis=2), axis=1).tolist()

    def to_variable(self, v):
        if self.use_cuda:
            v = v.cuda() # move to gpu
        return Variable(v)

    def make_pred(self, model, frames):
        # Upsample prediction to frame length (because we want prediction for each frame)
        return F.interpolate(model(frames), frames.size(2), mode="linear", align_corners=True)

    def fit(self,
            mode="rgb", # can be "rgb" or "flow"
            p_frame=None, # the path to load rgb or optical flow frames
            p_model=None, # the path to load the pretrained or previously self-trained model
            p_metadata_train="../data/metadata_train.json",
            p_metadata_validation="../data/metadata_validation.json"):

        self.log("="*60)
        self.log("="*60)
        self.log("Start training...")

        # Check
        if p_frame is None:
            self.log("Please specify p_frame, the path to load rgb or optical flow frames")

        # Set model
        model = self.set_model(mode, p_model)
        if model is None: return None

        # Load datasets
        metadata_path = {"train": p_metadata_train, "validation": p_metadata_validation}
        transform = {"train": None, "validation": None}
        if self.augment:
            transform = {"train": transforms.Compose([RandomCrop(224), RandomHorizontalFlip()]), "validation": None}
        dataloader = self.set_dataloader(metadata_path, p_frame, mode, transform)

        # Set optimizer
        optimizer = optim.SGD(model.parameters(), lr=self.init_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_sche= optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

        # Set logging format
        log_fm = "%s step: %d lr: %r loc_loss: %.4f cls_loss: %.4f loss: %.4f"

        # Train and validate
        steps = 0
        epochs = 0
        nspu = self.num_steps_per_update
        nspc = self.num_steps_per_check
        nspu_nspc = nspu * nspc
        model_id = str(uuid.uuid4())[0:7] + "-i3d-" + mode
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
                    model.train(True) # set model to training mode
                else:
                    model.train(False) # set model to evaluate mode
                optimizer.zero_grad()
                # Iterate over data
                for d in dataloader[phase]:
                    accum[phase] += 1
                    # Get prediction
                    frames = self.to_variable(d["frames"])
                    labels = d["labels"]
                    true_labels[phase] += self.labels_to_list(labels)
                    labels = self.to_variable(labels)
                    pred = self.make_pred(model, frames)
                    pred_labels[phase] += self.labels_to_list(pred.cpu().detach())
                    # Compute localization loss
                    #TODO: think about if we need the localization loss
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
                            tll = tot_loc_loss[phase]/nspu_nspc
                            tcl = tot_cls_loss[phase]/nspu_nspc
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
                    p_model = self.save_model_path + model_id + "/" + str(steps) + ".pt"
                    self.save(model, p_model)
                    for phase in ["train", "validation"]:
                        self.log("Performance for " + phase)
                        self.log(classification_report(true_labels[phase], pred_labels[phase]))
                        pred_labels[phase] = []
                        true_labels[phase] = []

        self.log("Done fit")

    def predict(self,
            mode="rgb", # can be "rgb" or "flow"
            p_frame=None, # the path to load rgb or optical flow frames
            p_model=None, # the path to load the previously self-trained model
            p_metadata_test="../data/metadata_test.json"):

        self.log("="*60)
        self.log("="*60)
        self.log("Start testing...")

        # Check
        if p_frame is None:
            self.log("Please specify p_frame, the path to load rgb or optical flow frames")

        # Set model
        model = self.set_model(mode, p_model)
        if model is None: return None

        # Load dataset
        metadata_path = {"test": p_metadata_test}
        transform = {"test": None}
        dataloader = self.set_dataloader(metadata_path, p_frame, mode, transform)

        # Test
        model.train(False)
        true_labels = []
        pred_labels = []
        counter = 0
        with torch.no_grad():
            for d in dataloader["test"]:
                if counter % 20 == 0:
                    self.log("Process batch " + str(counter))
                counter += 1
                frames = self.to_variable(d["frames"])
                labels = d["labels"]
                true_labels += self.labels_to_list(labels)
                labels = self.to_variable(labels)
                pred = self.make_pred(model, frames)
                pred_labels += self.labels_to_list(pred.cpu().detach())
        self.log(classification_report(true_labels, pred_labels))

        self.log("Done test")
