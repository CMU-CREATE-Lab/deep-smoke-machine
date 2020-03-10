import torch
import torch.nn as nn
from base_learner import BaseLearner
from model.cp_pytorch_ts import *
from model.pytorch_fuse import *
from torch.utils.data import DataLoader
from smoke_video_dataset_cp import SmokeVideoDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import uuid
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
import numpy as np
import random
from torchvision import transforms
from video_transforms import *
from torch.utils.tensorboard import SummaryWriter

import json

# Two-Stream ConvNet learner
# http://papers.nips.cc/paper/5353-two-stream-convolutional
class FuseLearner(BaseLearner):
    def __init__(self,
                 batch_size=32,
                 lr=0.1,
                 max_steps=64e3,
                 momentum=0.95,
                 #milestones = [40, 500, 1000],
                 gamma = 0.1,
                 weight_decay = 0.000001,
                 num_workers=1,
                 num_of_action_classes=2,
                 num_steps_per_update=1,
                 num_steps_per_check=10,
                 #use_cuda=torch.cuda.is_available(),
                 use_cuda=True,
                 parallel=True,
                 augment=True,
                 save_model_path="../data/saved_ts/",  # path for saving the models
                 train_writer_p="../data/ts_runs/ts_train",
                 val_writer_p="../data/ts_runs/ts_val"
                 ):
        super().__init__()
        self.create_logger(log_path="TsLearner.log")
        self.log("Use Two-Stream ConvNet learner")

        self.batch_size = batch_size
        self.lr = lr
        self.max_steps = max_steps
        self.momentum = momentum
        #self.milestones = milestones
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.num_of_action_classes = num_of_action_classes
        self.num_steps_per_update = num_steps_per_update
        self.num_steps_per_check = num_steps_per_check
        self.use_cuda = use_cuda
        self.parallel = parallel
        self.augment = augment
        self.save_model_path = save_model_path
        self.train_writer_p = train_writer_p
        self.val_writer_p = val_writer_p

    def random_frames_from_batch(self, data):
        b, c, f, h, w = list(data.shape)
        new_batch = torch.zeros(b, c, h, w)
        for i in range(b):
            frame = random.randint(0, f - 1)
            new_batch[i, :, :, :] = data[i, :, frame, :, :]
        return new_batch

    def compress_videos_to_frames(self, n):
        b, c, t, h, w = n.shape
        ret = torch.zeros(b, c*t, h, w)
        frame = 0
        for channel in range(c*t):
            ret[:,channel,:,:] = n[:,channel%c, frame,:,:]
            if channel > 0 and channel % c == 0: frame += 1
        return ret

    def set_dataloader(self, metadata_path, p_vid, mode, tf, sec_vid):
        dataloader = {}
        for phase in metadata_path:
            self.log("Create dataloader for " + phase)
            dataset = SmokeVideoDataset(metadata_path=metadata_path[phase], root_dir=p_vid, transform = tf[phase], second_dir=sec_vid) #mode=mode, transform=tf[phase])
            #dataloader[phase] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
            dataloader[phase] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return dataloader

    def labels_to_list(self, labels):
        # Convert labels obtained from the dataloader to a list of action classes
        return np.argmax(labels.numpy().max(axis=2), axis=1).tolist()

    def to_variable(self, v):
        if self.use_cuda:
            v = v.cuda()  # move to gpu
        return Variable(v)

    def make_pred(self, model, frames):
        # Upsample prediction to frame length (because we want prediction for each frame)
        return F.interpolate(model(frames), frames.size(2), mode="linear", align_corners=True)

    def fit(self,
            mode="rgb",
            p_metadata_train="../data/metadata_train.json",
            p_metadata_validation="../data/metadata_validation.json",
            p_vid="../data/"):

        self.log("="*60)
        self.log("="*60)
        self.log("Start training...")

        ts_rgb = SpatialCNN()
        ts_flow = MotionCNN()
        fuse = FusionNN()
        p_rgb = p_vid + "rgb/"
        p_flow = p_vid + "flow_old/"

        rgb_model = "../data/saved_ts/rgb/long_train/d0c1b2c-ts-rgb/39825.pt"
        flow_model = "../data/saved_ts/flow/first_train/2295.pt"

        self.load(ts_rgb, rgb_model)
        self.load(ts_flow, flow_model)

        if self.use_cuda:
            ts_rgb.cuda()
            ts_flow.cuda()
            fuse.cuda()
            if self.parallel and torch.cuda.device_count() > 1:
                self.log("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
                ts_rgb = nn.DataParallel(ts_rgb)
                ts_flow = nn.DataParallel(ts_flow)
                fuse = nn.DataParallel(fuse)

        writer_train = SummaryWriter(self.train_writer_p)
        writer_val = SummaryWriter(self.val_writer_p)

        # Load datasets
        metadata_path = {"train": p_metadata_train, "validation": p_metadata_validation}
        transform = None
        if self.augment:
            transform  = {"train": transforms.Compose([RandomCrop(224), RandomHorizontalFlip()]), "validation": None}
        dataloader = self.set_dataloader(metadata_path, p_rgb, "rgb", transform, p_flow)

        # Set optimizer
        optimizer = optim.SGD(params=fuse.parameters(), lr=self.lr, momentum=self.momentum, weight_decay = self.weight_decay)
        # lr_sche = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

        # Set Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # Set logging format
        log_fm = "%s step: %d lr: %r loss: %.4f"

        # Train and validate
        ts_rgb.train(False)
        ts_flow.train(False)
        steps = 0
        epochs = 0
        nspu = self.num_steps_per_update
        nspc = self.num_steps_per_check
        nspu_nspc = nspu * nspc
        model_id = str(uuid.uuid4())[0:7] + "-fuse-" + mode
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
                    fuse.train(True) # set model to training mode
                else:
                    fuse.train(False) # set model to evaluate mode
                optimizer.zero_grad()
                # Iterate over data
                for d in dataloader[phase]:
                    accum[phase] += 1
                    # Get inputs
                    frames_rgb, labels, frames_flow = d["frames"], d["labels"], d["second"]
                    # Set RGB frames
                    frames_rgb = self.compress_videos_to_frames(frames_rgb)
                    frames_rgb = self.to_variable(frames_rgb)
                    # Set Flow frames
                    frames_flow = self.compress_videos_to_frames(frames_flow)
                    frames_flow = self.to_variable(frames_flow)
                    # Set labels
                    l = self.labels_to_list(labels)
                    #true_labels += self.labels_to_list(labels)
                    true_labels[phase] += l
                    labels = self.to_variable(d["labels"])
                    pred_rgb = ts_rgb(frames_rgb)
                    pred_flow = ts_flow(frames_flow)
                    pred_both = torch.cat([pred_rgb, pred_flow], dim=1)
                    pred = fuse(pred_both)
                    _, predicted = torch.max(pred, dim=1)
                    pred_labels[phase] += list(predicted.cpu().detach())
                    # Compute classification loss (with max-pooling along time, batch x channel x time)
                    cls_loss = criterion(pred, torch.max(labels, dim=2)[0])
                    tot_cls_loss[phase] += cls_loss.data
                    # Backprop
                    loss = cls_loss / nspu
                    tot_loss[phase] += loss.data
                    loss.backward()
                    # Accumulate gradients during training
                    if (accum[phase] == nspu) and phase == "train":
                        steps += 1
                        accum[phase] = 0
                        optimizer.step()
                        optimizer.zero_grad()
                        # lr_sche.step()
                        if steps % nspc == 0:
                            #lr = lr_sche.get_lr()[0]
                            lr = self.lr
                            tl = tot_loss[phase]/nspc
                            writer_train.add_scalar("Training Loss", tl, global_step=steps)
                            self.log(log_fm % (phase, steps, lr, tl))
                            tot_loss[phase] = 0.0
                if phase == "validation":
                    # lr = lr_sche.get_lr()[0]
                    lr = self.lr
                    tl = (tot_loss[phase]*nspu)/accum[phase]
                    writer_val.add_scalar("Validation Loss", tl, global_step=steps)
                    self.log(log_fm % (phase, steps, lr, tl))
                    tot_loss[phase] = 0.0
                    accum[phase] = 0
                    p_model = self.save_model_path + model_id + "/" + str(steps) + ".pt"
                    self.save(fuse, p_model)
                    for phase in ["train", "validation"]:
                        self.log("Performance for " + phase)
                        accuracy = accuracy_score(true_labels[phase], pred_labels[phase])
                        if phase == "train": writer_train.add_scalar("Train Accuracy", accuracy, global_step=steps)
                        elif phase == "validation": writer_val.add_scalar("Val Accuracy", accuracy, global_step=steps)
                        self.log(classification_report(true_labels[phase], pred_labels[phase]))
                        pred_labels[phase] = []
                        true_labels[phase] = []

        self.log("Done fit")

    def test(self,
                p_metadata_test="../data/metadata_test.json",
                p_vid="../data/",
                p_model=None):
        self.log("="*60)
        self.log("="*60)
        self.log("Start testing...")

        ts_rgb = SpatialCNN()
        ts_flow = MotionCNN()
        fuse = FusionNN()
        p_rgb = p_vid + "rgb/"
        p_flow = p_vid + "flow/"

        self.load(ts_rgb, "../data/saved_ts/rgb/long_train/d0c1b2c-ts-rgb/39825.pt")
        self.load(ts_flow, "../data/saved_ts/flow/first_train/2295.pt")
        self.load(fuse, p_model)

        if self.use_cuda:
            ts_rgb.cuda()
            ts_flow.cuda()
            fuse.cuda()
            if self.parallel and torch.cuda.device_count() > 1:
                self.log("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
                ts_rgb = nn.DataParallel(ts_rgb)
                ts_flow = nn.DataParallel(ts_flow)
                fuse = nn.DataParallel(fuse)

        # Load dataset
        metadata_path = {"test": p_metadata_test}
        transform = {"test": None}
        dataloader = self.set_dataloader(metadata_path, p_rgb, "rgb", transform, p_flow)

        # Test
        ts_rgb.train(False)
        ts_flow.train(False)
        fuse.train(False)
        true_labels = []
        pred_labels = []
        ######
        all_pred = {}
        ######
        counter = 0
        with torch.no_grad():
            for d in dataloader["test"]:
                if counter % 20 == 0:
                    self.log("Process batch " + str(counter))
                counter += 1
                # Get inputs
                frames_rgb, labels, frames_flow = d["frames"], d["labels"], d["second"]
                # Set RGB frames
                frames_rgb = self.compress_videos_to_frames(frames_rgb)
                frames_rgb = self.to_variable(frames_rgb)
                # Set Flow frames
                frames_flow = self.compress_videos_to_frames(frames_flow)
                frames_flow = self.to_variable(frames_flow)
                # Set labels
                l = self.labels_to_list(labels)
                #true_labels += self.labels_to_list(labels)
                true_labels += l
                labels = self.to_variable(d["labels"])
                pred_rgb = ts_rgb(frames_rgb)
                pred_flow = ts_flow(frames_flow)
                pred_both = torch.cat([pred_rgb, pred_flow], dim=1)
                pred = fuse(pred_both)
                _, predicted = torch.max(pred, dim=1)
                pred_labels += list(predicted.cpu().detach())
        self.log(classification_report(true_labels, pred_labels))
        self.log(precision_recall_curve(true_labels, pred_labels))

        self.log("Done test")
        pass
