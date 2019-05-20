import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # use the order in the nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # specify which GPU(s) to be used
import torch
from torch.utils.data import DataLoader
from smoke_video_dataset import SmokeVideoDataset
from model.pytorch_i3d import InceptionI3d
from torch.autograd import Variable
import numpy as np
from util import *

def flatten_tensor(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t

# Extract I3D features from pre-trained models and save them
def main(argv):
    mode = "rgb"
    batch_size = 32
    p = "../data/"
    p_feat = p + "features/"
    p_pretrain = p + "pretrained_models/"
    p_vid = p + "videos/"
    has_gpu = torch.cuda.is_available()

    # Setup the model and load pre-trained weights
    if mode == "rgb":
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load(p_pretrain+"i3d_rgb_imagenet_kinetics.pt"))
    elif mode == "flow":
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(p_pretrain+"i3d_flow_imagenet_kinetics.pt"))
    else:
        return None
    i3d.replace_logits(157)

    # Use GPU or not
    if has_gpu:
        i3d.cuda()

    # Set the model to evaluation mode
    i3d.train(False)

    # Check the directory for saving features
    check_and_create_dir(p_feat)

    # Loop all datasets
    for phase in ["train", "validation", "test"]:
        print("Create dataset for", phase)
        dataset = SmokeVideoDataset(metadata_path=p+"metadata_"+phase+".json", root_dir=p_vid, mode=mode)
        print("Create dataloader for", phase)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # Iterate over data batches
        for d in dataloader:
            # Skip if all the files exist
            skip = True
            file_name = d["file_name"]
            for fn in file_name:
                if not is_file_here(p_feat+fn+".npy"):
                    skip = False
                    break
            if skip: continue
            # Extract features
            with torch.no_grad():
                frames = d["frames"]
                if has_gpu:
                    frames = frames.cuda()
                frames = Variable(frames)
                features = i3d.extract_features(frames)
            for i in range(len(file_name)):
                f = flatten_tensor(features[i, :, :, :, :])
                fn = file_name[i]
                print("Save feature", fn)
                np.save(os.path.join(p_feat, fn), f.data.cpu().numpy())
    print("Done")

if __name__ == "__main__":
    main(sys.argv)
