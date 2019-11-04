import os
import torch
from torch.utils.data import Dataset
import numpy as np
from util import *


# The smoke video dataset
class SmokeVideoDataset(Dataset):
    def __init__(self, metadata_path=None, root_dir=None, transform=None):
        """
        metadata_path (string): the full path to the video metadata json file
        root_dir (string): the root directory that stores video files
        transform (callable, optional): optional transform to be applied on a video
        """
        self.metadata = load_json(metadata_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        v = self.metadata[idx]

        # Load video data
        # TODO: convert video data to lmdb format (https://lmdb.readthedocs.io/en/release/)
        # TODO: in the __init__ function, open the lmdb, and in here, load the data by index
        file_path = os.path.join(self.root_dir, v["file_name"] + ".npy")
        if not is_file_here(file_path):
            raise ValueError("Cannot find file: %s" % (file_path))
        frames = np.load(file_path).astype(np.uint8)
        t = frames.shape[0]

        # Transform video
        if self.transform:
            frames = self.transform(frames)

        # Process labels
        label = v["label"]
        # TODO: change the [0.0, 1.0] to [0.0, weight]?
        if label == 1:
            labels = np.array([0.0, 1.0], dtype=np.float32) # The 2st column show the probability of yes
        else:
            labels = np.array([1.0, 0.0], dtype=np.float32) # The 1st column show the probability of no
        labels = np.repeat([labels], t, axis=0) # Repeat for each frame (frame by frame detection)

        return {"frames": frames, "labels": self.labels_to_tensor(labels), "file_name": v["file_name"]}

    def labels_to_tensor(self, labels):
        """
        Converts a numpy.ndarray with shape (time x num_of_action_classes)
        to a torch.FloatTensor of shape (num_of_action_classes x time)
        """
        return torch.from_numpy(labels.transpose([1,0]))


# The smoke video feature dataset
class SmokeVideoFeatureDataset(Dataset):
    def __init__(self, metadata_path=None, root_dir=None):
        """
        metadata_path (string): the full path to the video metadata json file
        root_dir (string): the root directory that stores video feature files
        """
        self.metadata = load_json(metadata_path)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        v = self.metadata[idx]

        # Load rgb or optical flow features as one data point
        feature_file_path = os.path.join(self.root_dir, v["file_name"] + ".npy")
        if not is_file_here(feature_file_path):
            raise ValueError("Cannot find file: %s" % (feature_file_path))
        feature = np.load(feature_file_path)

        # Return item
        return {"feature": feature, "label": v["label"], "file_name": v["file_name"]}
