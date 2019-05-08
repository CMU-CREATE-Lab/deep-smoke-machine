import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # use the order in the nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # specify which GPU(s) to be used

from torch.utils.data import Dataset, DataLoader
import cv2
from util import *

# The smoke video dataset
class SmokeVideoDataset(Dataset):
    def __init__(self, metadata_path, root_dir, transform=None):
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
        file_path = os.path.join(self.root_dir, self.metadata[idx]["file_name"])


# Extract features from pre-trained models and save them
def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)
