import torch
import torch.nn as nn
from base_learner import BaseLearner
from model.pytorch_ts import *
from torch.utils.data import DataLoader
from smoke_video_dataset import SmokeVideoDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import uuid
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import random
from torchvision.transforms import Compose
from video_transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomPerspective, RandomErasing, Resize, Normalize
from torch.utils.tensorboard import SummaryWriter
from util import *


# Two-Stream ConvNet learner
# http://papers.nips.cc/paper/5353-two-stream-convolutional
class TsLearner(BaseLearner):
    def __init__(self,
                 batch_size=32,
                 mode="rgb",
                 lr=0.1,
                 max_steps=64e3,
                 momentum=0.9,
                 milestones = [200, 400, 1000, 1800],
                 gamma=0.1,
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
        self.mode = mode
        self.lr = lr
        self.max_steps = max_steps
        self.momentum = momentum
        self.milestones = milestones
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

    def set_dataloader(self, metadata_path, p_vid, mode, tf):
        dataloader = {}
        for phase in metadata_path:
            self.log("Create dataloader for " + phase)
            dataset = SmokeVideoDataset(metadata_path=metadata_path[phase], root_dir=p_vid, transform = tf[phase]) #mode=mode, transform=tf[phase])
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
            p_metadata_train="../data/split/metadata_train_split_0_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_0_by_camera.json",
            p_vid="../data/"):

        self.log("="*60)
        self.log("="*60)
        self.log("Start training...")

        if self.mode == "rgb":
            ts = SpatialCNN()
            p_vid = p_vid + "rgb/"

        elif self.mode == "flow":
            ts = MotionCNN()
            p_vid = p_vid + "flow/"

        if self.use_cuda:
            ts.cuda()
            if self.parallel and torch.cuda.device_count() > 1:
                self.log("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
                ts = nn.DataParallel(ts)

        writer_train = SummaryWriter(self.train_writer_p)
        writer_val = SummaryWriter(self.val_writer_p)

        # Load datasets
        metadata_path = {"train": p_metadata_train, "validation": p_metadata_validation}
        tf = self.get_transform(self.mode)
        transform = {"train": tf, "validation": tf}
        if self.augment:
            transform["train"] = self.get_transform(phase="train", mode=self.mode)
        dataloader = self.set_dataloader(metadata_path, p_vid, self.mode, transform)

        # Set optimizer
        optimizer = optim.SGD(params=ts.parameters(), lr=self.lr, momentum=self.momentum, weight_decay = self.weight_decay)
        lr_sche = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

        # Set Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # Set logging format
        log_fm = "%s step: %d lr: %r loss: %.4f"

        # Train and validate
        steps = 0
        epochs = 0
        nspu = self.num_steps_per_update
        nspc = self.num_steps_per_check
        nspu_nspc = nspu * nspc
        model_id = str(uuid.uuid4())[0:7] + "-ts-" + self.mode
        accum = {} # counter for accumulating gradients
        tot_loss = {} # total loss
        tot_loc_loss = {} # total localization loss
        tot_cls_loss = {} # total classification loss
        pred_labels = {} # predicted labels
        true_labels = {} # true labels
        file_name = {}
        for phase in ["train", "validation"]:
            accum[phase] = 0
            tot_loss[phase] = 0.0
            tot_loc_loss[phase] = 0.0
            tot_cls_loss[phase] = 0.0
            pred_labels[phase] = []
            true_labels[phase] = []
            file_name[phase] = []
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
                #for d in dataloader[phase]:
                for count, d in enumerate(dataloader[phase]):
                    accum[phase] += 1
                    file_name[phase] += d["file_name"]
                    # Get inputs
                    frames, labels = d["frames"], d["labels"]
                    frames = self.compress_videos_to_frames(frames)
                    frames = self.to_variable(frames)
                    true_labels[phase] += self.labels_to_list(labels)
                    labels = self.to_variable(d["labels"])
                    pred = ts(frames)
                    _, predicted = torch.max(pred, dim=1)
                    pred_labels[phase] += predicted.cpu().detach().tolist()
                    #pred_labels[phase] += list(predicted.cpu().detach())
                    # Compute classification loss (with max-pooling along time, batch x channel x time)
                    cls_loss = criterion(pred, torch.max(labels, dim=2)[0])
                    tot_cls_loss[phase] += cls_loss.data
                    # Backprop
                    loss = cls_loss / nspu
                    tot_loss[phase] += loss.data
                    if phase != "validation":
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
                            tl = tot_loss[phase]/nspc
                            writer_train.add_scalar("Training Loss", tl, global_step=steps)
                            self.log(log_fm % (phase, steps, lr, tl))
                            tot_loss[phase] = 0.0
                if phase == "validation":
                    lr = lr_sche.get_lr()[0]
                    tl = (tot_loss[phase]*nspu)/accum[phase]
                    writer_val.add_scalar("Validation Loss", tl, global_step=steps)
                    self.log(log_fm % (phase, steps, lr, tl))
                    tot_loss[phase] = 0.0
                    accum[phase] = 0
                    p_model = self.save_model_path + model_id + "/" + str(steps) + ".pt"
                    self.save(ts, p_model)
                    for phase in ["train", "validation"]:
                        self.log("Performance for " + phase)
                        accuracy = accuracy_score(true_labels[phase], pred_labels[phase])
                        if phase == "train": writer_train.add_scalar("Train Accuracy", accuracy, global_step=steps)
                        elif phase == "validation": writer_val.add_scalar("Val Accuracy", accuracy, global_step=steps)
                        self.log(classification_report(true_labels[phase], pred_labels[phase]))

                        # Add video summary to tensorboard
                        cm = confusion_matrix_of_samples(true_labels[phase], pred_labels[phase], n=4)
                        print(cm)
                        writer = writer_train if phase == "train" else writer_val
                        write_video_summary(writer, cm, file_name[phase], p_vid, global_step=steps)
                        # Reset
                        pred_labels[phase] = []
                        true_labels[phase] = []
                        file_name[phase] = []

        # This is a hack to give the summary writer some time to write the data
        # Without this hack, the last added video will be missing
        time.sleep(10)

        self.log("Done fit")

    def test(self,
                p_metadata_test="../data/split/metadata_test_split_0_by_camera.json",
                p_vid="../data/",
                p_model=None):
        self.log("="*60)
        self.log("="*60)
        self.log("Start testing...")

        if self.mode == "rgb":
            ts = SpatialCNN()
            p_vid = p_vid + "rgb/"

        elif self.mode == "flow":
            ts = MotionCNN()
            p_vid = p_vid + "flow/"

        self.load(ts, p_model)

        if self.use_cuda:
            ts.cuda()
            if self.parallel and torch.cuda.device_count() > 1:
                self.log("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
                ##Uncomented below
                ts = nn.DataParallel(ts)

        # Load dataset
        metadata_path = {"test": p_metadata_test}
        transform = {"test": self.get_transform(self.mode)}
        dataloader = self.set_dataloader(metadata_path, p_vid, self.mode, transform)

        # Test
        ts.train(False)
        file_name = []
        true_labels = []
        pred_labels = []
        counter = 0
        with torch.no_grad():
            for d in dataloader["test"]:
                if counter % 20 == 0:
                    self.log("Process batch " + str(counter))
                counter += 1
                # Get inputs
                file_name += d["file_name"]
                frames, labels = d["frames"], d["labels"]
                frames = self.compress_videos_to_frames(frames)
                frames = self.to_variable(frames)
                true_labels += self.labels_to_list(labels)
                labels = self.to_variable(d["labels"])
                ##Added softmax below
                pred = F.softmax(ts(frames), dim=1)
                _, predicted = torch.max(pred, dim=1)
                pred_labels += list(predicted.cpu().detach())
        self.log(classification_report(true_labels, pred_labels))

        #TODO: setup tensorboard for test

        #cm = confusion_matrix_of_samples(true_labels, pred_labels)
        #write_video_summary(writer, cm, file_name, p_frame)

        self.log("Done test")
