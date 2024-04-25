import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # use the order in the nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # specify which GPU(s) to be used
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # specify the tensorflow log level
from base_learner import BaseLearner
from torch.utils.data import DataLoader
from smoke_video_dataset import SmokeVideoDataset
from model.pytorch_i3d import InceptionI3d
from model.pytorch_i3d_tc import InceptionI3dTc
from model.pytorch_i3d_tsm import InceptionI3dTsm
from model.pytorch_i3d_lstm import InceptionI3dLstm
from model.pytorch_i3d_nl import InceptionI3dNl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import uuid
from sklearn.metrics import classification_report as cr
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from util import *
import re
import time
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import shutil
from model.tsm.ops.models import TSN


# Two-Stream Inflated 3D ConvNet learner
# https://arxiv.org/abs/1705.07750
class I3dLearner(BaseLearner):
    def __init__(self,
            use_cuda=None, # use cuda or not
            use_tsm=False, # use the Temporal Shift module or not
            use_nl=False, # use the Non-local module or not
            use_tc=False, # use the Timeception module or not
            use_lstm=False, # use LSTM module or not
            freeze_i3d=False, # freeze i3d layers when training Timeception
            batch_size_train=10, # size for each batch for training
            batch_size_test=20, # size for each batch for testing
            batch_size_extract_features=40, # size for each batch for extracting features
            max_steps=2000, # total number of steps for training
            num_steps_per_update=2, # gradient accumulation (for large batch size that does not fit into memory)
            init_lr=0.1, # initial learning rate
            weight_decay=0.000001, # L2 regularization
            momentum=0.9, # SGD parameters
            milestones=[500, 1500], # MultiStepLR parameters
            gamma=0.1, # MultiStepLR parameters
            num_of_action_classes=2, # currently we only have two classes (0 and 1, which means no and yes)
            num_steps_per_check=50, # the number of steps to save a model and log information
            parallel=True, # use nn.DistributedDataParallel or not
            augment=True, # use data augmentation or not
            num_workers=12, # number of workers for the dataloader
            mode="rgb", # can be "rgb" or "flow" or "rgbd"
            p_frame="../data/rgb/", # path to load video frames
            code_testing=False # a special flag for testing if the code works
            ):
        super().__init__(use_cuda=use_cuda)

        self.use_tsm = use_tsm
        self.use_nl = use_nl
        self.use_tc = use_tc
        self.use_lstm = use_lstm
        self.freeze_i3d = freeze_i3d
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.batch_size_extract_features = batch_size_extract_features
        self.max_steps = max_steps
        self.num_steps_per_update = num_steps_per_update
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.milestones = milestones
        self.gamma = gamma
        self.num_of_action_classes = num_of_action_classes
        self.num_steps_per_check = num_steps_per_check
        self.parallel = parallel
        self.augment = augment
        self.num_workers = num_workers
        self.mode = mode
        self.p_frame = p_frame

        # Internal parameters
        self.image_size = 224 # 224 is the input for the i3d network structure
        self.can_parallel = False

        # Code testing mode
        self.code_testing = code_testing
        if code_testing:
            self.max_steps = 10

    def log_parameters(self):
        text = "\nParameters:\n"
        text += "  use_cuda: " + str(self.use_cuda) + "\n"
        text += "  use_tsm: " + str(self.use_tsm) + "\n"
        text += "  use_nl: " + str(self.use_nl) + "\n"
        text += "  use_tc: " + str(self.use_tc) + "\n"
        text += "  use_lstm: " + str(self.use_lstm) + "\n"
        text += "  freeze_i3d: " + str(self.freeze_i3d) + "\n"
        text += "  batch_size_train: " + str(self.batch_size_train) + "\n"
        text += "  batch_size_test: " + str(self.batch_size_test) + "\n"
        text += "  batch_size_extract_features: " + str(self.batch_size_extract_features) + "\n"
        text += "  max_steps: " + str(self.max_steps) + "\n"
        text += "  num_steps_per_update: " + str(self.num_steps_per_update) + "\n"
        text += "  init_lr: " + str(self.init_lr) + "\n"
        text += "  weight_decay: " + str(self.weight_decay) + "\n"
        text += "  momentum: " + str(self.momentum) + "\n"
        text += "  milestones: " + str(self.milestones) + "\n"
        text += "  gamma: " + str(self.gamma) + "\n"
        text += "  num_of_action_classes: " + str(self.num_of_action_classes) + "\n"
        text += "  num_steps_per_check: " + str(self.num_steps_per_check) + "\n"
        text += "  parallel: " + str(self.parallel) + "\n"
        text += "  augment: " + str(self.augment) + "\n"
        text += "  num_workers: " + str(self.num_workers) + "\n"
        text += "  mode: " + self.mode + "\n"
        text += "  p_frame: " + self.p_frame + "\n"
        self.log(text)

    def set_model(self, rank, world_size, mode, p_model, parallel, phase="train"):
        model_batch_size = self.batch_size_train
        if phase == "test":
            model_batch_size = self.batch_size_test
        elif phase == "feature":
            model_batch_size = self.batch_size_extract_features

        # Setup the model based on mode
        # The reason why we use 400 classes at the begining is because of loading the pretrained model
        has_extra_layers = self.use_tc or self.use_tsm or self.use_nl or self.use_lstm
        nc_kinetics = 400
        if mode == "rgb" or mode == "rgbd":
            ic = 3 if mode == "rgb" else 4
            if not has_extra_layers:
                model = InceptionI3d(num_classes=nc_kinetics, in_channels=ic)
            else:
                input_size = [model_batch_size, ic, 36, 224, 224] # (batch_size, channel, time, height, width)
                if self.use_tsm:
                    model = InceptionI3dTsm(input_size, num_classes=nc_kinetics, in_channels=ic)
                elif self.use_tc:
                    model = InceptionI3dTc(input_size, num_classes=nc_kinetics, in_channels=ic,
                            freeze_i3d=self.freeze_i3d)
                elif self.use_lstm:
                    model = InceptionI3dLstm(input_size, num_classes=nc_kinetics, in_channels=ic,
                            freeze_i3d=self.freeze_i3d)
                elif self.use_nl:
                    model = InceptionI3dNl(input_size, num_classes=nc_kinetics, in_channels=ic)
                else:
                    raise NotImplementedError("Not implemented.")
        elif mode == "flow":
            ic = 2
            if not has_extra_layers:
                model = InceptionI3d(num_classes=nc_kinetics, in_channels=ic)
            else:
                raise NotImplementedError("Not implemented.")
        else:
            return None

        # Try loading pre-trained i3d weights (from the 400-class model trained on the Kinetics dataset)
        error_1 = False
        try:
            self.log("Try loading pre-trained i3d weights (from the 400-class model trained on the Kinetics dataset)")
            if p_model is not None:
                if has_extra_layers:
                    self.load(model.get_i3d_model(), p_model, rank=rank)
                else:
                    if mode == "rgbd":
                        self.load(model, p_model, rank=rank, fill_dim=True)
                    else:
                        self.load(model, p_model, rank=rank)
        except Exception as e:
            self.log(e)
            # This means that the i3d weights are self-trained
            error_1 = True

        # Set the number of output classes
        # Note that for the TSM model this function is empty (no need to replace the last layer)
        model.replace_logits(self.num_of_action_classes)

        # Try loading self-trained weights (from the 2-class model fine-tuned on our dataset)
        error_2 = False
        try:
            self.log("Try loading self-trained weights (from the 2-class model fine-tuned on our dataset)")
            if error_1 and p_model is not None:
                if has_extra_layers:
                    self.load(model.get_i3d_model(), p_model, rank=rank)
                else:
                    self.load(model, p_model, rank=rank)
        except Exception as e:
            self.log(e)
            # This means that the model we want to load has extra layers
            error_2 = True

        # Delete the unused logits layers in the I3D model if using extra layers
        if has_extra_layers:
            model.delete_i3d_logits()

        # Try loading self-trained weights with extra layers
        if error_2 and p_model is not None:
            self.log("Try loading self-trained weights with extra layers")
            self.load(model, p_model, rank=rank)

        # Add TSM
        if self.use_tsm:
            model.add_tsm_to_i3d()

        # Add NL blocks
        if self.use_nl:
            model.add_nl_to_i3d()

        # Use GPU or not
        if self.use_cuda:
            if parallel:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                # Rank 1 means one machine, world_size means the number of GPUs on that machine
                dist.init_process_group("nccl", rank=rank, world_size=world_size)
                if p_model is None:
                    # Make sure that models on different GPUs start from the same initialized weights
                    torch.manual_seed(42)
                n = torch.cuda.device_count() // world_size
                device_ids = list(range(rank * n, (rank + 1) * n))
                torch.cuda.set_device(rank)
                model.cuda(rank)
                model = DDP(model.to(device_ids[0]), device_ids=device_ids)
            else:
                model.cuda()

        return model

    def set_dataloader(self, rank, world_size, metadata_path, root_dir, transform, batch_size, parallel):
        dataloader = {}
        for phase in metadata_path:
            self.log("Create dataloader for " + phase)
            dataset = SmokeVideoDataset(metadata_path=metadata_path[phase], root_dir=root_dir, transform=transform[phase])
            if parallel:
                sampler = DistributedSampler(dataset, shuffle=True, num_replicas=world_size, rank=rank)
                dataloader[phase] = DataLoader(dataset, batch_size=batch_size,
                        num_workers=int(self.num_workers/world_size), pin_memory=True, sampler=sampler)
            else:
                dataloader[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=self.num_workers, pin_memory=True)
        return dataloader

    def labels_to_list(self, labels):
        # Convert labels obtained from the dataloader to a list of action classes
        return np.argmax(labels.numpy().max(axis=2), axis=1).tolist()

    def labels_to_score_list(self, labels):
        # Convert labels obtained from the dataloader to a list of action class scores
        return labels.numpy().max(axis=2).tolist()

    def to_variable(self, v):
        if self.use_cuda:
            v = v.cuda() # move to gpu
        return v

    def make_pred(self, model, frames, upsample=True):
        m = model(frames)
        if upsample == True:
            # Upsample prediction to frame length (because we want prediction for each frame)
            return F.interpolate(m, frames.size(2), mode="linear", align_corners=True)
        elif upsample == False:
            return m
        else:
            # Return both results
            return (m, F.interpolate(m, frames.size(2), mode="linear", align_corners=True))

    def flatten_tensor(self, t):
        t = t.reshape(1, -1)
        t = t.squeeze()
        return t

    def clean_mp(self):
        if self.can_parallel:
            dist.destroy_process_group()

    def fit(self,
            p_model=None, # the path to load the pretrained or previously self-trained model
            model_id_suffix="", # the suffix appended after the model id
            p_metadata_train="../data/split/metadata_train_split_0_by_camera.json", # metadata path (train)
            p_metadata_validation="../data/split/metadata_validation_split_0_by_camera.json", # metadata path (validation)
            p_metadata_test="../data/split/metadata_test_split_0_by_camera.json", # metadata path (test)
            save_model_path="../data/saved_i3d/[model_id]/model/", # path to save the models ([model_id] will be replaced)
            save_tensorboard_path="../data/saved_i3d/[model_id]/run/", # path to save data ([model_id] will be replaced)
            save_log_path="../data/saved_i3d/[model_id]/log/train.log", # path to save log files ([model_id] will be replaced)
            save_metadata_path="../data/saved_i3d/[model_id]/metadata/" # path to save metadata ([model_id] will be replaced)
            ):
        # Set path
        model_id = str(uuid.uuid4())[0:7] + "-i3d-" + self.mode
        model_id += model_id_suffix
        save_model_path = save_model_path.replace("[model_id]", model_id)
        save_tensorboard_path = save_tensorboard_path.replace("[model_id]", model_id)
        save_log_path = save_log_path.replace("[model_id]", model_id)
        save_metadata_path = save_metadata_path.replace("[model_id]", model_id)

        # Copy training, validation, and testing metadata
        check_and_create_dir(save_metadata_path)
        shutil.copy(p_metadata_train, save_metadata_path + "metadata_train.json")
        shutil.copy(p_metadata_validation, save_metadata_path + "metadata_validation.json")
        shutil.copy(p_metadata_test, save_metadata_path + "metadata_test.json")

        # Spawn processes
        n_gpu = torch.cuda.device_count()
        if self.parallel and n_gpu > 1:
            self.can_parallel = True
            self.log("Let's use " + str(n_gpu) + " GPUs!")
            mp.spawn(self.fit_worker, nprocs=n_gpu,
                    args=(n_gpu, p_model, save_model_path, save_tensorboard_path, save_log_path, self.p_frame,
                        p_metadata_train, p_metadata_validation, p_metadata_test), join=True)
        else:
            self.fit_worker(0, 1, p_model, save_model_path, save_tensorboard_path, save_log_path, self.p_frame,
                    p_metadata_train, p_metadata_validation, p_metadata_test)

    def fit_worker(self, rank, world_size, p_model, save_model_path, save_tensorboard_path, save_log_path,
            p_frame, p_metadata_train, p_metadata_validation, p_metadata_test):
        # Set logger
        save_log_path += str(rank)
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use Two-Stream Inflated 3D ConvNet learner")
        self.log("save_model_path: " + save_model_path)
        self.log("save_tensorboard_path: " + save_tensorboard_path)
        self.log("save_log_path: " + save_log_path)
        self.log("p_metadata_train: " + p_metadata_train)
        self.log("p_metadata_validation: " + p_metadata_validation)
        self.log("p_metadata_test: " + p_metadata_test)
        self.log_parameters()

        # Set model
        model = self.set_model(rank, world_size, self.mode, p_model, self.can_parallel, phase="train")
        if model is None: return None

        # Load datasets
        metadata_path = {"train": p_metadata_train, "validation": p_metadata_validation}
        ts = self.get_transform(self.mode, image_size=self.image_size)
        transform = {"train": ts, "validation": ts}
        if self.augment:
            transform["train"] = self.get_transform(self.mode, phase="train", image_size=self.image_size)
        dataloader = self.set_dataloader(rank, world_size, metadata_path, p_frame,
                transform, self.batch_size_train, self.can_parallel)

        # Create tensorboard writter
        writer_t = SummaryWriter(save_tensorboard_path + "/train/")
        writer_v = SummaryWriter(save_tensorboard_path + "/validation/")

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
            # Each epoch has a training and validation phase
            for phase in ["train", "validation"]:
                self.log("-"*40)
                self.log("phase " + phase)
                if phase == "train":
                    epochs += 1
                    self.log("epochs: %d steps: %d/%d" % (epochs, steps, self.max_steps))
                    model.train(True) # set model to training mode
                    for param in model.parameters():
                        param.requires_grad = True
                else:
                    model.train(False) # set model to evaluate mode
                    for param in model.parameters():
                        param.requires_grad = False
                optimizer.zero_grad()
                # Iterate over batch data
                for d in tqdm.tqdm(dataloader[phase]):
                    if self.code_testing:
                        if phase == "train" and steps >= self.max_steps: break
                        if phase == "validation" and accum[phase] >= self.max_steps: break
                    accum[phase] += 1
                    # Get prediction
                    frames = self.to_variable(d["frames"])
                    labels = d["labels"]
                    true_labels[phase] += self.labels_to_list(labels)
                    labels = self.to_variable(labels)
                    pred = self.make_pred(model, frames)
                    pred_labels[phase] += self.labels_to_list(pred.cpu().detach())
                    # Compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(pred, labels)
                    tot_loc_loss[phase] += loc_loss.data
                    # Compute classification loss (with max-pooling along time, batch x channel x time)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(pred, dim=2)[0], torch.max(labels, dim=2)[0])
                    tot_cls_loss[phase] += cls_loss.data
                    # Backprop
                    loss = (0.5*loc_loss + 0.5*cls_loss) / nspu
                    tot_loss[phase] += loss.data
                    if phase == "train":
                        loss.backward()
                    # Accumulate gradients during training
                    if (accum[phase] == nspu) and phase == "train":
                        steps += 1
                        if steps % nspc == 0:
                            # Log learning rate and loss
                            lr = lr_sche.get_lr()[0]
                            tll = tot_loc_loss[phase]/nspu_nspc
                            tcl = tot_cls_loss[phase]/nspu_nspc
                            tl = tot_loss[phase]/nspc
                            self.log(log_fm % (phase, steps, lr, tll, tcl, tl))
                            # Add to tensorboard
                            if rank == 0:
                                writer_t.add_scalar("localization_loss", tll, global_step=steps)
                                writer_t.add_scalar("classification_loss", tcl, global_step=steps)
                                writer_t.add_scalar("loss", tl, global_step=steps)
                                writer_t.add_scalar("learning_rate", lr, global_step=steps)
                            # Reset loss
                            tot_loss[phase] = tot_loc_loss[phase] = tot_cls_loss[phase] = 0.0
                        # Reset gradient accumulation
                        accum[phase] = 0
                        # Update learning rate and optimizer
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_sche.step()
                    # END FOR LOOP
                if phase == "validation":
                    # Log learning rate and loss
                    lr = lr_sche.get_lr()[0]
                    tll = tot_loc_loss[phase]/accum[phase]
                    tcl = tot_cls_loss[phase]/accum[phase]
                    tl = (tot_loss[phase]*nspu)/accum[phase]
                    # Sync losses for validation set
                    if self.can_parallel:
                        tll_tcl_tl = torch.Tensor([tll, tcl, tl]).cuda()
                        dist.all_reduce(tll_tcl_tl, op=dist.ReduceOp.SUM)
                        tll = tll_tcl_tl[0].item() / world_size
                        tcl = tll_tcl_tl[1].item() / world_size
                        tl = tll_tcl_tl[2].item() / world_size
                    self.log(log_fm % (phase, steps, lr, tll, tcl, tl))
                    # Add to tensorboard and save model
                    if rank == 0:
                        writer_v.add_scalar("localization_loss", tll, global_step=steps)
                        writer_v.add_scalar("classification_loss", tcl, global_step=steps)
                        writer_v.add_scalar("loss", tl, global_step=steps)
                        writer_v.add_scalar("learning_rate", lr, global_step=steps)
                        self.save(model, save_model_path + str(steps) + ".pt")
                    # Reset loss
                    tot_loss[phase] = tot_loc_loss[phase] = tot_cls_loss[phase] = 0.0
                    # Reset gradient accumulation
                    accum[phase] = 0
                    # Save precision, recall, and f-score to the log and tensorboard
                    for ps in ["train", "validation"]:
                        # Sync true_labels and pred_labels for validation set
                        if self.can_parallel and ps == "validation":
                            true_pred_labels = torch.Tensor([true_labels[ps], pred_labels[ps]]).cuda()
                            true_pred_labels_list = [torch.ones_like(true_pred_labels) for _ in range(world_size)]
                            dist.all_gather(true_pred_labels_list, true_pred_labels)
                            true_pred_labels = torch.cat(true_pred_labels_list, dim=1)
                            true_labels[ps] = true_pred_labels[0].cpu().numpy()
                            pred_labels[ps] = true_pred_labels[1].cpu().numpy()
                        self.log("Evaluate performance of phase: %s\n%s" % (ps, cr(true_labels[ps], pred_labels[ps])))
                        if rank == 0:
                            result = prfs(true_labels[ps], pred_labels[ps], average="weighted")
                            writer = writer_t if ps == "train" else writer_v
                            writer.add_scalar("precision", result[0], global_step=steps)
                            writer.add_scalar("recall", result[1], global_step=steps)
                            writer.add_scalar("weighted_fscore", result[2], global_step=steps)
                        # Reset
                        pred_labels[ps] = []
                        true_labels[ps] = []

        # Clean processors
        self.clean_mp()

        self.log("Done training")

    def test(self,
            p_model=None # the path to load the pretrained or previously self-trained model
            ):
        # Check
        if p_model is None or not is_file_here(p_model):
            self.log("Need to provide a valid model path")
            return

        # Set path
        match = re.search(r'\b/[0-9a-fA-F]{7}-i3d-(rgb|flow)[^/]*/\b', p_model)
        model_id = match.group()[1:-1]
        if model_id is None:
            self.log("Cannot find a valid model id from the model path.")
            return
        p_root = p_model[:match.start()] + "/" + model_id + "/"
        p_metadata_test = p_root + "metadata/metadata_test.json" # metadata path (test)
        save_log_path = p_root + "log/test.log" # path to save log files
        save_viz_path = p_root + "viz/" # path to save visualizations

        # Spawn processes
        n_gpu = torch.cuda.device_count()
        if False:#self.parallel and n_gpu > 1:
            # TODO: multiple GPUs will cause an error when generating summary videos
            self.can_parallel = True
            self.log("Let's use " + str(n_gpu) + " GPUs!")
            mp.spawn(self.test_worker, nprocs=n_gpu,
                    args=(n_gpu, p_model, save_log_path, self.p_frame, save_viz_path, p_metadata_test), join=True)
        else:
            self.test_worker(0, 1, p_model, save_log_path, self.p_frame, save_viz_path, p_metadata_test)

    def test_worker(self, rank, world_size, p_model, save_log_path, p_frame, save_viz_path, p_metadata_test):
        # Set logger
        save_log_path += str(rank)
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use Two-Stream Inflated 3D ConvNet learner")
        self.log("Start testing with mode: " + self.mode)
        self.log("save_log_path: " + save_log_path)
        self.log("save_viz_path: " + save_viz_path)
        self.log("p_metadata_test: " + p_metadata_test)
        self.log_parameters()

        # Set model
        model = self.set_model(rank, world_size, self.mode, p_model, self.can_parallel, phase="test")
        if model is None: return None

        # Load dataset
        metadata_path = {"test": p_metadata_test}
        transform = {"test": self.get_transform(self.mode, image_size=self.image_size)}
        dataloader = self.set_dataloader(rank, world_size, metadata_path, p_frame,
                transform, self.batch_size_test, self.can_parallel)

        # Test
        model.train(False) # set the model to evaluation mode
        file_name = []
        true_labels = []
        pred_labels = []
        true_scores = []
        pred_scores = []
        counter = 0
        with torch.no_grad():
            # Iterate over batch data
            for d in dataloader["test"]:
                if counter % 5 == 0:
                    self.log("Process batch " + str(counter))
                counter += 1
                file_name += d["file_name"]
                frames = self.to_variable(d["frames"])
                labels = d["labels"]
                true_labels += self.labels_to_list(labels)
                true_scores += self.labels_to_score_list(labels)
                labels = self.to_variable(labels)
                pred = self.make_pred(model, frames)
                pred = pred.cpu().detach()
                pred_labels += self.labels_to_list(pred)
                pred_scores += self.labels_to_score_list(pred)

        # Sync true_labels and pred_labels for testing set
        true_labels_all = np.array(true_labels)
        pred_labels_all = np.array(pred_labels)
        true_scores_all = np.array(true_scores)
        pred_scores_all = np.array(pred_scores)

        if self.can_parallel:
            true_pred_labels = torch.Tensor([true_labels, pred_labels, true_scores, pred_scores]).cuda()
            true_pred_labels_list = [torch.ones_like(true_pred_labels) for _ in range(world_size)]
            dist.all_gather(true_pred_labels_list, true_pred_labels)
            true_pred_labels = torch.cat(true_pred_labels_list, dim=1)
            true_labels_all = true_pred_labels[0].cpu().numpy()
            pred_labels_all = true_pred_labels[1].cpu().numpy()
            true_scores_all = true_pred_labels[2].cpu().numpy()
            pred_scores_all = true_pred_labels[3].cpu().numpy()

        # Save precision, recall, and f-score to the log
        self.log("Evaluate performance of phase: test\n%s" % (cr(true_labels_all, pred_labels_all)))

        # Save roc curve and score
        self.log("roc_auc_score: %s" % str(roc_auc_score(true_scores_all, pred_scores_all, average=None)))

        # Generate video summary and show class activation map
        # TODO: this part will cause an error when using multiple GPUs
        try:
            # Video summary
            cm = confusion_matrix_of_samples(true_labels, pred_labels, n=64)
            write_video_summary(cm, file_name, p_frame, save_viz_path + str(rank) + "/")
            # Save confusion matrix
            cm_all = confusion_matrix_of_samples(true_labels, pred_labels)
            for u in cm_all:
                for v in cm_all[u]:
                    for i in range(len(cm_all[u][v])):
                        idx = cm_all[u][v][i]
                        cm_all[u][v][i] = file_name[idx]
            save_json(cm_all, save_viz_path + str(rank) + "/confusion_matrix_of_samples.json")
        except Exception as ex:
            self.log(ex)

        # Clean processors
        self.clean_mp()

        self.log("Done testing")

    def extract_features(self,
            p_model=None, # the path to load the pretrained or previously self-trained model
            p_feat="../data/i3d_features_rgb/", # path to save features
            p_metadata_train="../data/split/metadata_train_split_0_by_camera.json", # metadata path (train)
            p_metadata_validation="../data/split/metadata_validation_split_0_by_camera.json", # metadata path (validation)
            p_metadata_test="../data/split/metadata_test_split_0_by_camera.json", # metadata path (test)
            ):
        # Set path
        check_and_create_dir(p_feat) # check the directory for saving features

        # Log
        self.log("="*60)
        self.log("="*60)
        self.log("Use Two-Stream Inflated 3D ConvNet learner")
        self.log("Start extracting features...")

        # Set model (currently we use only one GPU for extracting features)
        model = self.set_model(0, 1, self.mode, p_model, False, phase="feature")
        if model is None: return None

        # Load datasets
        metadata_path = {"train": p_metadata_train, "validation": p_metadata_validation, "test": p_metadata_test}
        ts = self.get_transform(self.mode, image_size=self.image_size)
        transform = {"train": ts, "validation": ts, "test": ts}
        dataloader = self.set_dataloader(0, 1, metadata_path, self.p_frame,
                transform, self.batch_size_extract_features, False)

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
