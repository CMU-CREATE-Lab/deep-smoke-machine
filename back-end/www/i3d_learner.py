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
import uuid
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from util import *
import re
import time
import tqdm


# Two-Stream Inflated 3D ConvNet learner
# https://arxiv.org/abs/1705.07750
class I3dLearner(BaseLearner):
    def __init__(self,
            batch_size_train=32, # size for each batch for training (8 max for each GTX 1080Ti)
            batch_size_test=128, # size for each batch for testing (32 max for each GTX 1080Ti)
            batch_size_extract_features=32, # size for each batch for extracting features
            max_steps=16000, # total number of steps for training
            num_steps_per_update=2, # gradient accumulation (for large batch size that does not fit into memory)
            init_lr_rgb=0.1, # initial learning rate (for i3d-rgb)
            init_lr_flow=0.1, # initial learning rate (for i3d-flow)
            #weight_decay=0.0000001, # L2 regularization (for i3d-rgb)
            weight_decay=0.0000001, # L2 regularization (for i3d-flow)
            momentum=0.9, # SGD parameters
            milestones_rgb=[500, 1000, 2000, 4000], # MultiStepLR parameters (for i3d-rgb)
            milestones_flow=[500, 1000, 2000, 4000], # MultiStepLR parameters (for i3d-flow)
            gamma=0.1, # MultiStepLR parameters
            num_of_action_classes=2, # currently we only have two classes (0 and 1, which means no and yes)
            num_steps_per_check=100, # the number of steps to save a model and log information
            parallel=True, # use nn.DataParallel or not
            augment=True, # use data augmentation or not
            num_workers=12, # number of workers for the dataloader
            mode="rgb", # can be "rgb" or "flow"
            p_frame_rgb="../data/rgb/", # path to load rgb frame
            p_frame_flow="../data/flow/", # path to load optical flow frame
            p_metadata_train="../data/metadata_train.json", # path to load metadata for training
            p_metadata_validation="../data/metadata_validation.json", # path to load metadata for validation
            p_metadata_test="../data/metadata_test.json" # path to load metadata for testing
            ):
        super().__init__()

        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.batch_size_extract_features = batch_size_extract_features
        self.max_steps = max_steps
        self.num_steps_per_update = num_steps_per_update
        self.init_lr_rgb = init_lr_rgb
        self.init_lr_flow = init_lr_flow
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.milestones_rgb = milestones_rgb
        self.milestones_flow = milestones_flow
        self.gamma = gamma
        self.num_of_action_classes = num_of_action_classes
        self.num_steps_per_check = num_steps_per_check
        self.parallel = parallel
        self.augment = augment
        self.num_workers = num_workers
        self.mode = mode
        self.p_frame_rgb = p_frame_rgb
        self.p_frame_flow = p_frame_flow
        self.p_metadata_train = p_metadata_train
        self.p_metadata_validation = p_metadata_validation
        self.p_metadata_test = p_metadata_test
        self.image_size = 224 # 224 is the input for the i3d network structure

    def log_parameters(self):
        text = ""
        text += "batch_size_train: " + str(self.batch_size_train) + "\n"
        text += "batch_size_test: " + str(self.batch_size_test) + "\n"
        text += "batch_size_extract_features: " + str(self.batch_size_extract_features) + "\n"
        text += "max_steps: " + str(self.max_steps) + "\n"
        text += "num_steps_per_update: " + str(self.num_steps_per_update) + "\n"
        text += "init_lr_rgb: " + str(self.init_lr_rgb) + "\n"
        text += "init_lr_flow: " + str(self.init_lr_flow) + "\n"
        text += "weight_decay: " + str(self.weight_decay) + "\n"
        text += "momentum: " + str(self.momentum) + "\n"
        text += "milestones_rgb: " + str(self.milestones_rgb) + "\n"
        text += "milestones_flow: " + str(self.milestones_flow) + "\n"
        text += "gamma: " + str(self.gamma) + "\n"
        text += "num_of_action_classes: " + str(self.num_of_action_classes) + "\n"
        text += "num_steps_per_check: " + str(self.num_steps_per_check) + "\n"
        text += "parallel: " + str(self.parallel) + "\n"
        text += "augment: " + str(self.augment) + "\n"
        text += "num_workers: " + str(self.num_workers) + "\n"
        text += "mode: " + self.mode + "\n"
        text += "p_frame_rgb: " + self.p_frame_rgb + "\n"
        text += "p_frame_flow: " + self.p_frame_flow + "\n"
        text += "p_metadata_train: " + self.p_metadata_train + "\n"
        text += "p_metadata_validation: " + self.p_metadata_validation + "\n"
        text += "p_metadata_test: " + self.p_metadata_test
        self.log(text)

    def set_model(self, mode, p_model, parallel):
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
            if p_model is not None:
                self.load(model, p_model)
        except:
            # Not pre-trained weights
            is_self_trained = True

        # Set the number of output classes in the model
        model.replace_logits(self.num_of_action_classes)

        # Load self-trained weights from fine-tuned models
        if p_model is not None and is_self_trained:
            self.load(model, p_model)

        # Use GPU or not
        if self.use_cuda:
            model.cuda()
            if parallel and torch.cuda.device_count() > 1:
                self.log("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
                # TODO: change this to DistributedDataParallel, which is significantly faster
                model = nn.DataParallel(model)

        return model

    def set_dataloader(self, metadata_path, root_dir, transform, batch_size):
        dataloader = {}
        for phase in metadata_path:
            self.log("Create dataloader for " + phase)
            dataset = SmokeVideoDataset(metadata_path=metadata_path[phase], root_dir=root_dir, transform=transform[phase])
            dataloader[phase] = DataLoader(dataset, batch_size=batch_size,
                    shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return dataloader

    def labels_to_list(self, labels):
        # Convert labels obtained from the dataloader to a list of action classes
        return np.argmax(labels.numpy().max(axis=2), axis=1).tolist()

    def to_variable(self, v):
        if self.use_cuda:
            v = v.cuda() # move to gpu
        # TODO: in newer pytorch, Variable is no longer needed
        return Variable(v)

    def make_pred(self, model, frames):
        # Upsample prediction to frame length (because we want prediction for each frame)
        return F.interpolate(model(frames), frames.size(2), mode="linear", align_corners=True)

    def flatten_tensor(self, t):
        t = t.reshape(1, -1)
        t = t.squeeze()
        return t

    def fit(self,
            p_model=None, # the path to load the pretrained or previously self-trained model
            save_model_path="../data/saved_i3d/[model_id]/model/", # path to save the models ([model_id] will be replaced)
            save_tensorboard_path="../data/saved_i3d/[model_id]/run/", # path to save data ([model_id] will be replaced)
            save_log_path="../data/saved_i3d/[model_id]/log/train.log" # path to save log files ([model_id] will be replaced)
            ):
        # Set path
        model_id = str(uuid.uuid4())[0:7] + "-i3d-" + self.mode
        save_model_path = save_model_path.replace("[model_id]", model_id)
        save_tensorboard_path = save_tensorboard_path.replace("[model_id]", model_id)
        save_log_path = save_log_path.replace("[model_id]", model_id)
        p_frame = self.p_frame_rgb if self.mode == "rgb" else self.p_frame_flow

        # Set logger
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use Two-Stream Inflated 3D ConvNet learner")
        self.log("Start training model: " + model_id)
        self.log("save_model_path: " + save_model_path)
        self.log("save_tensorboard_path: " + save_tensorboard_path)
        self.log("save_log_path: " + save_log_path)
        self.log_parameters()

        # Set model
        model = self.set_model(self.mode, p_model, self.parallel)
        if model is None: return None

        # Load datasets
        metadata_path = {"train": self.p_metadata_train, "validation": self.p_metadata_validation}
        ts = self.get_transform(self.mode)
        transform = {"train": ts, "validation": ts}
        if self.augment:
            transform["train"] = self.get_transform(self.mode, phase="train")
        dataloader = self.set_dataloader(metadata_path, p_frame, transform, self.batch_size_train)

        # Create tensorboard writter
        writer_t = SummaryWriter(save_tensorboard_path + "/train/")
        writer_v = SummaryWriter(save_tensorboard_path + "/validation/")

        # Set optimizer
        init_lr = self.init_lr_rgb if self.mode == "rgb" else self.init_lr_flow
        optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        milestones = self.milestones_rgb if self.mode == "rgb" else self.milestones_flow
        lr_sche= optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=self.gamma)

        # Set logging format
        log_fm = "%s step: %d lr: %r loc_loss: %.4f cls_loss: %.4f loss: %.4f"

        # Train and validate
        steps = 0
        epochs = 0
        nspu = self.num_steps_per_update
        nspc = self.num_steps_per_check
        nspu_nspc = nspu * nspc
        accum = {} # counter for accumulating gradients
        tot_loss = {} # total loss
        tot_loc_loss = {} # total localization loss
        tot_cls_loss = {} # total classification loss
        pred_labels = {} # predicted labels
        true_labels = {} # true labels
        file_name = {} # file names of the labels
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
                    model.train(True) # set model to training mode
                else:
                    model.train(False) # set model to evaluate mode
                optimizer.zero_grad()
                # Iterate over batch data
                for d in tqdm.tqdm(dataloader[phase]):
                    continue
                    accum[phase] += 1
                    file_name[phase] += d["file_name"]
                    # Get prediction
                    frames = self.to_variable(d["frames"])
                    labels = d["labels"]
                    true_labels[phase] += self.labels_to_list(labels)
                    labels = self.to_variable(labels)
                    pred = self.make_pred(model, frames)
                    pred_labels[phase] += self.labels_to_list(pred.cpu().detach())
                    # Compute localization loss (TODO: think about if we need the localization loss)
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
                        if steps % nspc == 0:
                            # Save learning rate and loss to the log and tensorboard
                            lr = lr_sche.get_lr()[0]
                            tll = tot_loc_loss[phase]/nspu_nspc
                            tcl = tot_cls_loss[phase]/nspu_nspc
                            tl = tot_loss[phase]/nspc
                            writer_t.add_scalar("localization_loss", tll, global_step=steps)
                            writer_t.add_scalar("classification_loss", tcl, global_step=steps)
                            writer_t.add_scalar("loss", tl, global_step=steps)
                            writer_t.add_scalar("learning_rate", lr, global_step=steps)
                            self.log(log_fm % (phase, steps, lr, tll, tcl, tl))
                            # Reset
                            tot_loss[phase] = tot_loc_loss[phase] = tot_cls_loss[phase] = 0.0
                        # Reset
                        accum[phase] = 0
                        # Update learning rate and optimizer
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_sche.step()
                if phase == "validation":
                    # Save learning rate and loss to the log and tensorboard
                    lr = lr_sche.get_lr()[0]
                    tll = tot_loc_loss[phase]/accum[phase]
                    tcl = tot_cls_loss[phase]/accum[phase]
                    tl = (tot_loss[phase]*nspu)/accum[phase]
                    writer_v.add_scalar("localization_loss", tll, global_step=steps)
                    writer_v.add_scalar("classification_loss", tcl, global_step=steps)
                    writer_v.add_scalar("loss", tl, global_step=steps)
                    writer_v.add_scalar("learning_rate", lr, global_step=steps)
                    self.log(log_fm % (phase, steps, lr, tll, tcl, tl))
                    # Reset
                    tot_loss[phase] = tot_loc_loss[phase] = tot_cls_loss[phase] = 0.0
                    accum[phase] = 0
                    # Save model
                    self.save(model, save_model_path + str(steps) + ".pt")
                    for ps in ["train", "validation"]:
                        # Save precision, recall, and f-score to the log and tensorboard
                        prfs = precision_recall_fscore_support(true_labels[ps], pred_labels[ps], average="weighted")
                        writer = writer_t if ps == "train" else writer_v
                        writer.add_scalar("precision", prfs[0], global_step=steps)
                        writer.add_scalar("recall", prfs[1], global_step=steps)
                        writer.add_scalar("weighted_fscore", prfs[2], global_step=steps)
                        self.log("Performance for " + ps)
                        self.log(classification_report(true_labels[ps], pred_labels[ps]))
                        # Add video summary to tensorboard
                        cm = confusion_matrix_of_samples(true_labels[ps], pred_labels[ps])
                        write_video_summary(writer, cm, file_name[ps], p_frame, global_step=steps)
                        # Reset
                        pred_labels[ps] = []
                        true_labels[ps] = []
                        file_name[ps] = []

        # This is a hack to give the summary writer some time to write the data
        # Without this hack, the last added video will be missing
        time.sleep(10)

        self.log("Done training")

    def predict(self,
            p_model=None, # the path to load the pretrained or previously self-trained model
            save_tensorboard_path="../data/saved_i3d/[model_id]/run/", # path to save data ([model_id] will be replaced)
            save_log_path="../data/saved_i3d/[model_id]/log/test.log" # path to save log files ([model_id] will be replaced)
            ):
        # Check
        if p_model is None:
            self.log("Need to provide model path")
            return

        # Set path
        model_id = re.search(r'\b/[0-9a-fA-F]{7}-i3d-(rgb|flow)/\b', p_model).group()[1:-1]
        if model_id is None:
            model_id = "unknown-model-id"
        save_tensorboard_path = save_tensorboard_path.replace("[model_id]", model_id)
        save_log_path = save_log_path.replace("[model_id]", model_id)
        p_frame = self.p_frame_rgb if self.mode == "rgb" else self.p_frame_flow

        # Set logger
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use Two-Stream Inflated 3D ConvNet learner")
        self.log("Start testing with mode: " + self.mode)
        self.log("save_tensorboard_path: " + save_tensorboard_path)
        self.log("save_log_path: " + save_log_path)

        # Set model
        model = self.set_model(self.mode, p_model, self.parallel)
        if model is None: return None

        # Load dataset
        metadata_path = {"test": self.p_metadata_test}
        transform = {"test": self.get_transform(self.mode)}
        dataloader = self.set_dataloader(metadata_path, p_frame, transform, self.batch_size_test)

        # Create tensorboard writter
        writer = SummaryWriter(save_tensorboard_path + "/test/")

        # Test
        model.train(False) # set the model to evaluation mode
        file_name = []
        true_labels = []
        pred_labels = []
        counter = 0
        with torch.no_grad():
            # Iterate over batch data
            for d in dataloader["test"]:
                if counter % 20 == 0:
                    self.log("Process batch " + str(counter))
                counter += 1
                file_name += d["file_name"]
                frames = self.to_variable(d["frames"])
                labels = d["labels"]
                true_labels += self.labels_to_list(labels)
                labels = self.to_variable(labels)
                pred = self.make_pred(model, frames)
                pred_labels += self.labels_to_list(pred.cpu().detach())

        # Save precision, recall, and f-score to the log
        self.log(classification_report(true_labels, pred_labels))

        # Add video summary to tensorboard
        cm = confusion_matrix_of_samples(true_labels, pred_labels)
        write_video_summary(writer, cm, file_name, p_frame)

        # This is a hack to give the summary writer some time to write the data
        # Without this hack, the last added video will be missing
        time.sleep(10)

        self.log("Done testing")

    def extract_features(self,
            p_model=None, # the path to load the pretrained or previously self-trained model
            p_feat_rgb="../data/i3d_features_rgb/", # path to save rgb features
            p_feat_flow="../data/i3d_features_flow/" # path to save flow features
            ):
        # Set path
        p_frame = self.p_frame_rgb if self.mode == "rgb" else self.p_frame_flow
        p_feat = p_feat_rgb if self.mode == "rgb" else p_feat_flow
        check_and_create_dir(p_feat) # check the directory for saving features

        # Log
        self.log("="*60)
        self.log("="*60)
        self.log("Use Two-Stream Inflated 3D ConvNet learner")
        self.log("Start extracting features...")

        # Set model
        model = self.set_model(self.mode, p_model, False)
        if model is None: return None

        # Load datasets
        metadata_path = {"train": self.p_metadata_train,
            "validation": self.p_metadata_validation, "test": self.p_metadata_test}
        ts = self.get_transform(self.mode)
        transform = {"train": ts, "validation": ts, "test": ts}
        dataloader = self.set_dataloader(metadata_path, p_frame, transform, self.batch_size_extract_features)

        # Extract features
        model.train(False) # set the model to evaluation mode
        for phase in ["train", "validation", "test"]:
            self.log("phase " + phase)
            counter = 0
            # Iterate over batch data
            for d in dataloader[phase]:
                counter += 1
                # Skip if all the files in this batch exist
                skip = True
                file_name = d["file_name"]
                for fn in file_name:
                    if not is_file_here(p_feat + fn + ".npy"):
                        skip = False
                        break
                if skip:
                    self.log("Skip " + phase + " batch " + str(counter))
                    continue
                # Compute features
                with torch.no_grad():
                    frames = self.to_variable(d["frames"])
                    features = model.extract_features(frames)
                for i in range(len(file_name)):
                    f = self.flatten_tensor(features[i, :, :, :, :])
                    fn = file_name[i]
                    self.log("Save " + self.mode + " feature " + fn + ".npy")
                    np.save(os.path.join(p_feat, fn), f.data.cpu().numpy())

        self.log("Done extracting features")
