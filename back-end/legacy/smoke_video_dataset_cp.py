import torch
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
from util import *

# The smoke video dataset
class SmokeVideoDataset(Dataset):
    def __init__(self, metadata_path=None, root_dir=None, transform=None, second_dir=None):
        """
        metadata_path (string): the full path to the video metadata json file
        root_dir (string): the root directory that stores video files
        transform (callable, optional): optional transform to be applied on a video
        second (string): the root directory that stores the secondary video files for two stream network
        """
        self.metadata = load_json(metadata_path)
        self.root_dir = root_dir
        self.transform = transform
        self.second_dir = second_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        v = self.metadata[idx]

        # Load rgb or optical flow frames as one data point
        file_path = os.path.join(self.root_dir, v["file_name"] + ".npy")
        if not is_file_here(file_path):
            raise ValueError("Cannot find file: %s" % (file_path))
        frames = load_frames(file_path)

        if self.second_dir is not None:
            second = True
            second_path = os.path.join(self.second_dir, v["file_name"] + ".npy")
            if not is_file_here(second_path):
                raise ValueError("Cannot find file: %s" % (second_path))
            second_frames = load_frames(second_path)
        else:
            second = False

        # Transform the data point (e.g., data augmentation)
        if self.transform:
            frames = self.transform(frames)
            if second:
                second_frames = self.transform(second_frames)

        # Process label state to labels
        label_state = v["label_state_admin"] # TODO: need to change this to label_state in the future
        pos = [47, 23, 19, 15]
        neg = [32, 20, 16, 12]
        labels = np.array([0.0, 0.0], dtype=np.float32)

        # The 1st and 2nd column in the labels array show the probability of no and yes respectively
        if label_state in pos:
            labels[1] = 1.0
        elif label_state in neg:
            labels[0] = 1.0

        # Duplicate the label for each frame (frame by frame detection)
        labels = np.repeat([labels], frames.shape[0], axis=0)

        # Return item
        if not second:
            return {"frames": frames_to_tensor(frames), "labels": labels_to_tensor(labels), "file_name": v["file_name"]}
        else:
            return {"frames": frames_to_tensor(frames), "second": frames_to_tensor(second_frames), "labels": labels_to_tensor(labels), "file_name": v["file_name"]}

# Load preprocessed videos from file_path
def load_frames(file_path, resize_to=224.0):
    # Saved numpy files should be read in with format (time, height, width, channel)
    frames = np.load(file_path)
    t, h, w, c = frames.shape

    # Resize and scale images for the network structure
    #TODO: maybe use opencv to normalize the image
    #frames = cv.normalize(frames, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    frames_out = []
    need_resize = False
    if w < resize_to or h < resize_to:
        d = resize_to - min(w, h)
        sc = 1 + d / min(w, h)
        need_resize = True
    for i in range(t):
        img = frames[i, :, :, :]
        if need_resize:
            img = cv.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames_out.append(img)
    return np.asarray(frames_out, dtype=np.float32)

def labels_to_tensor(labels):
    """
    Converts a numpy.ndarray with shape (time x num_of_action_classes)
    to a torch.FloatTensor of shape (num_of_action_classes x time)
    """
    return torch.from_numpy(labels.transpose([1,0]))

def frames_to_tensor(frames):
    """
    Converts a numpy.ndarray with shape (time x height x width x channel)
    to a torch.FloatTensor of shape (channel x time x height x width)
    """
    return torch.from_numpy(frames.transpose([3,0,1,2]))

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
