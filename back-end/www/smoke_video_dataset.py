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

        # TODO: transform data to lmdb format (https://lmdb.readthedocs.io/en/release/)
        # TODO: in the __init__ function, open the lmdb, and in here, load the data by index
        # Load all data to memory
        print("Loading all data to memory...")
        self.video_data = []
        self.label_data = []
        pos = [47, 23, 19, 15]
        neg = [32, 20, 16, 12]
        for i in range(len(self.metadata)):
            v = self.metadata[i]
            # Add video data
            file_path = os.path.join(self.root_dir, v["file_name"] + ".npy")
            if not is_file_here(file_path):
                raise ValueError("Cannot find file: %s" % (file_path))
            frames = np.load(file_path).astype(np.uint8)
            self.video_data.append(frames)
            # Process label state to labels
            label_state = v["label_state_admin"] # TODO: need to change this to label_state in the future
            labels = np.array([0.0, 0.0], dtype=np.float32)
            if label_state in pos:
                labels[1] = 1.0 # The 2st column show the probability of yes
            elif label_state in neg:
                labels[0] = 1.0 # The 1st column show the probability of no
            labels = np.repeat([labels], frames.shape[0], axis=0) # Repeat for each frame (frame by frame detection)
            self.label_data.append(self.labels_to_tensor(labels))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        v = self.metadata[idx]
        frames = self.video_data[idx]
        if self.transform:
            frames = self.transform(frames)
        return {"frames": frames, "labels": self.label_data[idx], "file_name": v["file_name"]}

    def labels_to_tensor(labels):
        """
        Converts a numpy.ndarray with shape (time x num_of_action_classes)
        to a torch.FloatTensor of shape (num_of_action_classes x time)
        """
        return torch.from_numpy(labels.transpose([1,0]))


# The smoke video feature dataset
# TODO: improve performance of this class according to SmokeVideoDataset
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

        # Process label state to labels
        label_state = v["label_state_admin"] # TODO: need to change this to label_state in the future
        pos = [47, 23, 19, 15]
        neg = [32, 20, 16, 12]
        label = None
        if label_state in pos:
            label = 1
        elif label_state in neg:
            label = 0

        # Return item
        return {"feature": feature, "label": label, "file_name": v["file_name"]}
