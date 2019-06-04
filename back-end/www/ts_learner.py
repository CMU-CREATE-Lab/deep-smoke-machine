import torch
import torch.nn as nn
from base_learner import BaseLearner
from model.pytorch_ts import *
from torch.utils.data import DataLoader
from smoke_video_dataset import SmokeVideoDataset
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import uuid
from sklearn.metrics import classification_report
import numpy as np
import random


# Two-Stream ConvNet learner
# http://papers.nips.cc/paper/5353-two-stream-convolutional
class TsLearner(BaseLearner):
    def __init__(self,
                 batch_size=4,
                 lr=0.001,
                 max_steps=64e3,
                 momentum=0.9,
                 num_workers=1,
                 num_of_action_classes=2,
                 num_steps_per_update=3,
                 num_steps_per_check=10,
                 use_cuda = torch.cuda.is_available(),
                 parallel=False,
                 save_model_path="saved_ts/",  # path for saving the models
                 ):
        super().__init__()
        self.create_logger(log_path="TsLearner.log")
        self.log("Use Two-Stream ConvNet learner")

        self.batch_size = batch_size
        self.lr = lr
        self.max_steps = max_steps
        self.momentum = momentum
        self.num_workers = num_workers
        self.num_of_action_classes = num_of_action_classes
        self.num_steps_per_update = num_steps_per_update
        self.num_steps_per_check = num_steps_per_check
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.parallel = parallel
        self.save_model_path = save_model_path

    def random_frames_from_batch(self, data):
        b, c, f, h, w = list(data.shape)
        new_batch = torch.zeros(b, c, h, w)
        for i in range(b):
            frame = random.randint(0, f - 1)
            new_batch[i, :, :, :] = data[i, :, frame, :, :]
        return new_batch


    def set_dataloader(self, metadata_path, p_vid, mode):
        dataloader = {}
        for phase in metadata_path:
            self.log("Create dataloader for " + phase)
            dataset = SmokeVideoDataset(metadata_path=metadata_path[phase], root_dir=p_vid, mode=mode)
            dataloader[phase] = DataLoader(dataset, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.num_workers, pin_memory=True)

        return dataloader

    def labels_to_list(self, labels):
        # Convert labels obtained from the dataloader to a list of action classes
        return np.argmax(labels.numpy().max(axis=2), axis=1).tolist()

    def to_variable(self, v):
        if self.use_cuda:
            v = v.to(self.device) # move to gpu
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

        if mode == "rgb":
            ts = SpatialCNN()
            p_vid = p_vid + "rgb/"

        elif mode == "flow":
            ts = MotionCNN()
            p_vid = p_vid + "flow/"

        ts = ts.to(self.device)

        # Load datasets
        metadata_path = {"train": p_metadata_train, "validation": p_metadata_validation}
        dataloader = self.set_dataloader(metadata_path, p_vid, mode)

        # Set optimizer
        optimizer = optim.SGD(params=ts.parameters(), lr=self.lr, momentum=self.momentum)

        # Set Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # Set logging format
        log_fm = "%s step: %d lr: %r loc_loss: %.4f cls_loss: %.4f loss: %.4f"

        # Train and validate
        steps = 0
        epochs = 0
        nspu = self.num_steps_per_update
        nspc = self.num_steps_per_check
        nspu_nspc = nspu * nspc
        model_id = str(uuid.uuid4())[0:7] + "-ts-" + mode
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
                    ts.train(True) # set model to training mode
                else:
                    ts.train(False) # set model to evaluate mode
                optimizer.zero_grad()
                # Iterate over data
                for d in dataloader[phase]:
                    accum[phase] += 1
                    # Get inputs
                    frames, labels = d["frames"], d["labels"]
                    frames = self.random_frames_from_batch(frames)
                    frames = self.to_variable(frames)
                    true_labels[phase] += self.labels_to_list(labels)
                    labels = self.to_variable(d["labels"])
                    pred = ts(frames)
                    pred_labels[phase] += list(pred.cpu().detach())
                    # Compute localization loss
                    loc_loss = criterion(pred, torch.max(labels, dim=2)[0])
                    tot_loc_loss[phase] += loc_loss.data
                    # Compute classification loss (with max-pooling along time, batch x channel x time)
                    cls_loss = criterion(pred, torch.max(labels, dim=2)[0])
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
                        if steps % nspc == 0:
                            tll = tot_loc_loss[phase]/nspu_nspc
                            tcl = tot_cls_loss[phase]/nspu_nspc
                            tl = tot_loss[phase]/nspc
                            self.log(log_fm % (phase, steps, self.lr, tll, tcl, tl))
                            tot_loss[phase] = tot_loc_loss[phase] = tot_cls_loss[phase] = 0.0
                if phase == "validation":
                    tll = tot_loc_loss[phase]/accum[phase]
                    tcl = tot_cls_loss[phase]/accum[phase]
                    tl = (tot_loss[phase]*nspu)/accum[phase]
                    self.log(log_fm % (phase, steps, self.lr, tll, tcl, tl))
                    tot_loss[phase] = tot_loc_loss[phase] = tot_cls_loss[phase] = 0.0
                    accum[phase] = 0
                    p_model = self.save_model_path + model_id + "-" + str(steps) + ".pt"
                    self.save(ts, p_model)
                    for phase in ["train", "validation"]:
                        self.log("Performance for " + phase)
                        self.log(classification_report(true_labels[phase], pred_labels[phase]))
                        pred_labels[phase] = []
                        true_labels[phase] = []

        self.log("Done fit")

    def predict(self,
                mode="rgb",
                p_metadata_test="../data/metadata_test.json",
                p_vid="../data/videos/",
                p_model=None):
        self.log("="*60)
        self.log("="*60)
        self.log("Start testing...")


        # Load dataset
        metadata_path = {"test": p_metadata_test}
        dataloader = self.set_dataloader(metadata_path, p_vid, mode)

        if mode == "rgb":
            ts = SpatialCNN()

        elif mode == "flow":
            ts = MotionCNN()

        self.load(ts, p_model)

        # Test
        ts.train(False)
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
                pred = self.make_pred(ts, frames)
                pred_labels += self.labels_to_list(pred.cpu().detach())
        self.log(classification_report(true_labels, pred_labels))

        self.log("Done test")
        pass
