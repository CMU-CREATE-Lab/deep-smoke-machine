# deep-smoke-machine
Deep learning models and dataset for recognizing industrial smoke emissions. The videos are from the [smoke labeling tool](https://github.com/CMU-CREATE-Lab/video-labeling-tool). The code in this repository assumes that Ubuntu 18.04 server is installed. The code is released under the BSD 3-clause license, and the dataset is released under the Creative Commons Zero (CC0) license. If you found this dataset and the code useful, we would greatly appreciate it if you could cite our paper below:

Yen-Chia Hsu, Ting-Hao (Kenneth) Huang, Ting-Yao Hu, Paul Dille, Sean Prendi, Ryan Hoffman, Anastasia Tsuhlares, Jessica Pachuta, Randy Sargent, and Illah Nourbakhsh. 2021. Project RISE: Recognizing Industrial Smoke Emissions. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2021). https://ojs.aaai.org/index.php/AAAI/article/view/17739

> [!WARNING]
> There was an error in implementing the non-local blocks when we wrote the paper. We are very sorry about this error. The problem has been fixed in the code in this repository. However, the result of model RGB-NL in Table 7 in the paper is incorrect. We are working on re-running the code and will submit a corrected version of the paper to arXiv.

![This figure shows different types of videos (high-opacity smoke, low-opacity smoke, steam, and steam with smoke).](back-end/data/dataset/2020-02-24/smoke-type.gif)

The following figures show how the [I3D model](https://arxiv.org/abs/1705.07750) recognizes industrial smoke. The heatmaps (red and yellow areas on top of the images) indicate where the model thinks have smoke emissions. The examples are from the testing set with different camera views, which means that the model never sees these views at the training stage. These visualizations are generated by using the [Grad-CAM](https://arxiv.org/abs/1610.02391) technique. The x-axis indicates time.

![Example of the smoke recognition result.](back-end/data/dataset/2020-02-24/0-1-2019-01-17-6007-928-6509-1430-180-180-3906-1547732890-1547733065-grad-cam.png)

![Example of the smoke recognition result.](back-end/data/dataset/2020-02-24/0-7-2019-01-11-3544-899-4026-1381-180-180-7891-1547236155-1547236330-grad-cam.png)

![Example of the smoke recognition result.](back-end/data/dataset/2020-02-24/1-0-2018-08-24-3018-478-3536-996-180-180-8732-1535140050-1535140315-grad-cam.png)

### Table of Content
- [Install NVIDIA drivers, CUDA, and cuDNN](#install-nvidia)
- [Setup this tool](#setup-tool)
- [Use this tool](#use-this-tool)
- [Code infrastructure](#code-infrastructure)
- [Dataset](#dataset)
- [Pretrained models](#pretrained-models)
- [Deploy models to recognize smoke](#deploy-models-to-recognize-smoke)
- [Administrator only tasks](#admin)
- [Acknowledgements](#acknowledgements)

# <a name="install-nvidia"></a>Install NVIDIA drivers, CUDA, and cuDNN

This installation guide assusmes that you are using the Ubuntu 22.04 server version (not desktop version) operating system.

## Disable the nouveau driver

Run the following on open the file (assuming that you use `vim` as the text editor):
```sh
sudo vim /etc/modprobe.d/blacklist.conf
```

Then, add the following to the file to blacklist nouveau driver:
```sh
# Blacklist nouveau driver (for NVIDIA driver installation)
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```

Next, regenerate the initial ram file system of the kernel and reboot the computer:
```sh
sudo update-initramfs -u
sudo reboot now
```

Then, check if nouveau is disabled correctly using the following commands. You should not see any outputs from the terminal when running these commands.
```sh
lsmod | grep -i nouveau
dpkg -l | grep -i nouveau
```

## Install CUDA and the NVIDIA driver
We need to remove old NVIDIA drivers before installing the new one. For drivers that are installed using the files that are downloaded from NVIDIA website, run the following:
```sh
# For drivers that are installed from NVIDIA website file, remove the driver using the following command:
sudo nvidia-uninstall
```

For drivers that are installed using `sudo apt-get`, run the folloing:
```sh
# For drivers that are installed using sudo apt-get, , remove the driver using the following commands:
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get autoremove
```

We also need to make sure that the following packages are installed:
```sh
sudo apt-get install build-essential
sudo apt-get install linux-headers-$(uname -r)
```

Then, we can install the NVIDIA drivers while installing [CUDA](https://developer.nvidia.com/cuda-toolkit). Different versions of CUDA can be found [here](https://developer.nvidia.com/cuda-toolkit-archive). We use CUDA 11.8 for this project.
```sh
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

Follow the instruction on the terminal to install CUDA. Once it is done, run the following commands to add the CUDA runtime library.
```sh
sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig
```

Then, open the `.bashrc` file and add the CUDA paths:
```sh
vim ~/.bashrc

# Add the following lines to the file
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Next, reboot the machine:
```sh
sudo reboot now
```

Check if NVIDIA driver is installed. You should see something on the terminal when running the following commands.
```sh
sudo nvidia-smi
lsmod | grep -i nvidia
lspci | grep -i nvidia
```

Optinoally, you can also check CUDA installation by running the following command.
```sh
nvcc -V
```

Finally, install cuDNN. Documentation can be found [here](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html). Visit [the NVIDIA cuDNN page](https://developer.nvidia.com/cudnn) to download and install cuDNN. Below are the commands from the cuDNN downloading link:
```sh
wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-11
```

# <a name="setup-tool"></a>Setup this tool

This guide generally assumes that the Ubuntu 22.04 server version (not desktop version) operating system is installed.

## Install miniconda on Ubuntu

A detailed documentation is [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). First visit [here](https://conda.io/miniconda.html) to obtain the miniconda downloading path. The following script install conda for all users to the `/opt/miniconda3` directory:
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
```

Then, add conda to the bashrc file so that all users can use the conda command.
```sh
sudo vim /etc/bash.bashrc
# Add the following lines to this file
export PATH="/opt/miniconda3/bin:$PATH"
. /opt/miniconda3/etc/profile.d/conda.sh
```

After that, run the following (or exit the terminal and open a new terminal).
```sh
source /etc/bash.bashrc
```

## Install miniconda on Mac OS

For Mac OS, I recommend installing conda by using [Homebrew](https://brew.sh/).
```sh
brew cask install miniconda
echo 'export PATH="/usr/local/Caskroom/miniconda/base/bin:$PATH"' >> ~/.bash_profile
echo '. /usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh' >> ~/.bash_profile
source ~/.bash_profile
```

## Clone repository and create the conda environment

Clone this repository and set the permission.
```sh
git clone --recursive https://github.com/CMU-CREATE-Lab/deep-smoke-machine.git
sudo chown -R $USER deep-smoke-machine/
sudo chgrp -R $USER deep-smoke-machine/
sudo chmod -R 775 deep-smoke-machine/
```

For git to ignore permission changes.
```sh
# For only this repository
git config core.fileMode false

# For globally
git config --global core.fileMode false
```

Create conda environment and install packages.
```sh
conda create -n deep-smoke-machine
conda activate deep-smoke-machine
conda install python=3.9
conda install pip
which pip # make sure this is the pip inside the deep-smoke-machine environment
```

Install PyTorch by checking the command on the [PyTorch website](https://pytorch.org). An example for Ubuntu is below:
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install packages.
```sh
sh deep-smoke-machine/back-end/install_packages.sh
```

Update the optical_flow submodule.
```sh
cd deep-smoke-machine/back-end/www/optical_flow/
git submodule update --init --recursive
git checkout master
```

Install system packages for OpenCV.
```sh
sudo apt update
sudo apt install -y libsm6 libxext6 libxrender-dev
```

# <a name="use-this-tool"></a>Use this tool

To use our publicly released dataset (a snapshot of the [smoke labeling tool](http://smoke.createlab.org/) on 2/24/2020), we include [metadata_02242020.json](back-end/data/dataset/2020-02-24/metadata_02242020.json) file under the deep-smoke-machine/back-end/data/dataset/ folder. You need to copy, move, and rename this file to deep-smoke-machine/back-end/data/metadata.json.
```sh
cd deep-smoke-machine/back-end/data/
cp dataset/2020-02-24/metadata_02242020.json metadata.json
```

Split the metadata into three sets: train, validation, and test. This will create a deep-smoke-machine/back-end/data/split/ folder that contains all splits, as indicated in our paper. The method for splitting the dataset will be explained in the next "Dataset" section.
```sh
python split_metadata.py confirm
```

Download all videos in the metadata file to deep-smoke-machine/back-end/data/videos/. This will take a very long time, and we recommend running the code on the background using the [screen command](https://www.gnu.org/software/screen/manual/html_node/index.html).
```sh
python download_videos.py
```

> [!IMPORTANT]
>You ALWAYS need to deactivate the conda environment before running the `screen` command using `conda deactivate`. When you are inside a screen session, activate your conda environment again to run your code. If you forgot the deactivate conda before you enter a screen session, your code may not run correctly and can give errors.

Here are some tips for the screen command:
```sh
# List currently running screen session names
screen -ls

# Create a new screen session
screen
# Inside the screen, type "exit" to terminate the screen
# Use CTRL+C to interrupt a command
# Or use CTRL+A+D to detach the screen and send it to the background

# Go into a screen session, below is an example
# sudo screen -x 33186.download_videos
screen -x SCREEN_SESSION_NAME

# Terminate a screen session, below is an example
# sudo screen -X 33186.download_videos quit
screen -X SCREEN_SESSION_NAME quit

# Keep looking at the screen log
tail -f screenlog.0
```

Process and save all videos into RGB frames (under deep-smoke-machine/back-end/data/rgb/) and optical flow frames (under deep-smoke-machine/back-end/data/flow/). Because computing optical flow takes a very long time, by default, this script will only process RGB frames. If you need the optical flow frames, change the flow_type to 1 in the [`process_videos.py`](back-end/www/process_videos.py) script. The optical flow is only used for the flow-based models in the paper (e.g., `i3d-flow`, `svm-flow`).
```sh
python process_videos.py
```

Optionally, extract [I3D features](https://github.com/piergiaj/pytorch-i3d) under `deep-smoke-machine/back-end/data/i3d_features_rgb/` and `deep-smoke-machine/back-end/data/i3d_features_flow/`. This step is for the SVM models to work in the paper. You do not need to do this if you are not going to use the SVM model (e.g., `svm-rgb`, `svm-flow`).
```sh
python extract_features.py METHOD OPTIONAL_MODEL_PATH

# Extract features from pretrained i3d
python extract_features.py i3d-rgb
python extract_features.py i3d-flow

# Extract features from a saved i3d model
python extract_features.py i3d-rgb ../data/saved_i3d/ecf7308-i3d-rgb/model/16875.pt
python extract_features.py i3d-flow ../data/saved_i3d/af00751-i3d-flow/model/30060.pt
```

Train the model with cross-validation on all dataset splits, using different hyper-parameters. The model will be trained on the training set and validated on the validation set. Pretrained weights are obtained from the [pytorch-i3d repository](https://github.com/piergiaj/pytorch-i3d). By default, the information of the trained I3D model will be placed in the `deep-smoke-machine/back-end/data/saved_i3d/` folder. For the description of the models, please refer to our paper. Note that by default the PyTorch [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) GPU parallel computing is enabled (see [i3d_learner.py](back-end/www/i3d_learner.py)).
```sh
python train.py METHOD OPTIONAL_MODEL_PATH

# Use Two-Stream Inflated 3D ConvNet
python train.py i3d-rgb-cv-1

# Use I3D features + SVM
python train.py svm-rgb-cv-1
```

Test the performance of a model on the test set. This step will also generate summary videos for each cell in the confusion matrix (true positive, true negative, false positive, and false negative).
```sh
python test.py METHOD OPTIONAL_MODEL_PATH

# Use Two-Stream Inflated 3D ConvNet
python test.py i3d-rgb-cv-1 ../data/saved_i3d/ecf7308-i3d-rgb/model/16875.pt

# Use I3D features + SVM
python test.py svm-rgb-cv-1 ../data/saved_svm/445cc62-svm-rgb/model/model.pkl
```

Run [Grad-CAM](https://arxiv.org/abs/1610.02391) to visualize the areas in the videos that the model is looking at.
```sh
python grad_cam_viz.py i3d-rgb MODEL_PATH
```

After model training and testing, the folder structure will look like the following:
```sh
└── saved_i3d                            # this corresponds to deep-smoke-machine/back-end/data/saved_i3d/
    └── 549f8df-i3d-rgb-s1               # the name of the model, s1 means split 1
        ├── cam                          # the visualization using Grad-CAM
        ├── log                          # the log when training models
        ├── metadata                     # the metadata of the dataset split
        ├── model                        # the saved models
        ├── run                          # the saved information for TensorBoard
        └── viz                          # the sampled videos for each cell in the confusion matrix
```

If you want to see the training and testing results on [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html), run the following and go to the stated URL in your browser.
```sh
cd deep-smoke-machine/back-end/data/
tensorboard --logdir=saved_i3d
```

Recommended training strategy:
1. Set an initial learning rate (e.g., 0.1)
2. Keep this learning rate and train the model until the training error decreases too slow (or fluctuate) or until the validation error increases (a sign of overfitting)
3. Decrease the learning rate (e.g., by a factor of 10)
4. Load the best model weight from the ones that were trained using the previous learning rate
5. Repeat step 2, 3, and 4 until convergence

# <a name="code-structure"></a>Code infrastructure

This section explains the code infrastructure related to the I3D model training and testing in the [deep-smoke-machine/back-end/www/](back-end/www/) folder. Later in this section, I will describe how to build your own model and integrate it with the current pipeline. This code assumes that you are familiar with the [PyTorch deep learning framework](https://pytorch.org/). If you do not know PyTorch, I recommend checking [their tutorial page](https://pytorch.org/tutorials/) first.
- [base_learner.py](back-end/www/base_learner.py)
  - The abstract class for creating model learners. You will need to implement the fit and test function. This script provides shared functions, such as model loading, model saving, data augmentation, and progress logging.
- [i3d_learner.py](back-end/www/i3d_learner.py)
  - This script inherits the base_learner.py script for training the I3D models. This script contains code for back-propagation (e.g., loss function, learning rate scheduler, video batch loading) and GPU parallel computing (PyTorch DistributedDataParallel).
- [check_models.py](back-end/www/check_models.py)
  - Check if a developed model runs in simple cases. This script is used for debugging when developing new models.
- [smoke_video_dataset.py](back-end/www/smoke_video_dataset.py)
  - Definition of the dataset. This script inherits the PyTorch Dataset class for creating the DataLoader, which can be used to provide batches iteratively when training the models.
- [opencv_functional.py](back-end/www/opencv_functional.py)
  - A special utility function that mimics [torchvision.transforms.functional](https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html), designed for processing video frames and augmenting video data.
- [video_transforms.py](back-end/www/video_transforms.py)
  - A special utility function that mimics [torchvision.transforms.transforms](https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html), designed for processing video frames and augmenting video data.
- [deep-smoke-machine/back-end/www/model/](back-end/www/model/)
  - The place to put all models (e.g., I3D, Non-Local modules, Timeception modules, Temporal Shift modules, LSTM).

If you want to develop your own model, here are the steps that I recommend.
1. Play with the check_models.py script to understand the input and output dimensions.
2. Create your own model and place it in the deep-smoke-machine/back-end/www/model/ folder. You can take a look at other models to get an idea about how to write the code.
3. Import your model to the check_models.py script, then run the script to debug your model.
4. If you need a specific data augmentation pipeline, edit the get_transform function in the base_learner.py file. Depending on your needs, you may also need to edit the opencv_functional.py and video_transforms.py files.
5. Copy the i3d_learner.py file, import your model, and modify the code to suit your needs. Make sure that you import your customized learner class in the train.py and test.py files.

# <a name="dataset"></a>Dataset

We include our publicly released dataset (a snapshot of the [smoke labeling tool](http://smoke.createlab.org/) on 2/24/2020) [metadata_02242020.json](back-end/data/dataset/2020-02-24/metadata_02242020.json) file under the deep-smoke-machine/back-end/data/dataset/ folder. The JSON file contains an array, with each element in the array representing the metadata for a video. Each element is a dictionary with keys and values, explained below:
- camera_id
  - ID of the camera (0 means [clairton1](http://mon.createlab.org/#v=3703.5,970,0.61,pts&t=456.42&ps=25&d=2020-04-06&s=clairton1&bt=20200406&et=20200407), 1 means [braddock1](http://mon.createlab.org/#v=2868.5,740.5,0.61,pts&t=540.67&ps=25&d=2020-04-07&s=braddock1&bt=20200407&et=20200408), and 2 means [westmifflin1](http://mon.createlab.org/#v=1722.89321,1348.42994,0.806,pts&t=704.33&ps=25&d=2020-04-07&s=westmifflin1&bt=20200407&et=20200408))
- view_id
  - ID of the cropped view from the camera
  - Each camera produces a panarama, and each view is cropped from this panarama (will be explained later in this section)
- id
  - Unique ID of the video clip
- label_state
  - State of the video label produced by the citizen science volunteers (will be explained later in this section)
- label_state_admin
  - State of the video label produced by the researchers (will be explained later in this section)
- start_time
  - Starting epoch time (in seconds) when capturing the video, corresponding to the real-world time
- url_root
  - URL root of the video, need to combine with url_part to get the full URL (url_root + url_part)
- url_part
  - URL part of the video, need to combine with url_root to get the full URL (url_root + url_part)
- file_name
  - File name of the video, for example 0-1-2018-12-13-6007-928-6509-1430-180-180-6614-1544720610-1544720785
  - The format of the file_name is [camera_id]-[view_id]-[year]-[month]-[day]-[bound_left]-[bound_top]-[bound_right]-[bound_bottom]-[video_height]-[video_width]-[start_frame_number]-[start_epoch_time]-[end_epoch_time]
  - bound_left, bound_top, bound_right, and bound_bottom mean the bounding box of the video clip in the panarama

Note that the url_root and url_part point to videos with 180 by 180 resolutions. We also provide a higher resolution (320 by 320) version of the videos. Replace the "/180/" with "/320/" in the url_root, and also replace the "-180-180-" with "-320-320-" in the url_part. For example, see the following:
- URL for the 180 by 180 version: https://smoke.createlab.org/videos/180/2019-06-24/0-7/0-7-2019-06-24-3504-1067-4125-1688-180-180-9722-1561410615-1561410790.mp4
- URL for the 320 by 320 version: https://smoke.createlab.org/videos/320/2019-06-24/0-7/0-7-2019-06-24-3504-1067-4125-1688-320-320-9722-1561410615-1561410790.mp4

Each video is reviewed by at least two citizen science volunteers (or one researcher who received the [smoke reading training](https://www.eta-is-opacity.com/resources/method-9/)). Our paper describes the details of the labeling and quality control mechanism. The state of the label (label_state and label_state_admin) in the metadata_02242020.json is briefly explained below.
- 47 : gold standard positive
  - The researcher assigned a positive label to the video and indicated that the video should be used as a gold standard for data quality checks.
- 32 : gold standard negative
  - The researcher assigned a negative label to the video and indicated that the video should be used as a gold standard for data quality checks.
- 23 : strong positive
  - Two volunteers both agree (or one researcher says) that the video has smoke.
- 16 : strong negative
  - Two volunteers both agree (or one researcher says) that the video does not have smoke.
- 19 : weak positive
  - Two volunteers have different answers, and the third volunteer says that the video has smoke.
- 20 : weak negative
  - Two volunteers have different answers, and the third volunteer says that the video does not have smoke.
- 5 : maybe positive
  - One volunteers says that the video has smoke.
- 4 : maybe negative
  - One volunteers says that the video does not have smoke.
- 3 : has discord
  - Two volunteers have different answers (one says yes, and another one says no).
- -1 : no data, no discord
  - No data. If label_state_admin is -1, it means that the label is produced solely by citizen science volunteers. If label_state is -1, it means that the label is produced solely by researchers. Otherwise, the label is jointly produced by both citizen science volunteers and researchers. Please refer to our paper about these three cases.
- -2 : bad videos
  - This means that reseachers have checked the data and marked the video as not suitable for labeling (e.g., due to bad data quality such as incorrect image stitching or artifacts during video compression). These bad videos should not be used in building the model.

After running the [split_metadata.py](back-end/www/split_metadata.py) script, the "label_state" and "label_state_admin" keys in the dictionary will be aggregated into the final label, represented by the new "label" key (see the JSON files in the generated deep-smoke-machine/back-end/data/split/ folder). Positive (value 1) and negative (value 0) labels mean if the video clip has smoke emissions or not, respectively.

Also, the dataset will be divided into several splits, based on camera views or dates. The file names (without ".json" file extension) are listed below. The Split S<sub>0</sub>, S<sub>1</sub>, S<sub>2</sub>, S<sub>3</sub>, S<sub>4</sub>, and S<sub>5</sub> correspond to the ones indicated in the paper.

| Split | Train | Validate | Test |
| --- | --- | --- | --- |
| S<sub>0</sub> | metadata_train_split_0_by_camera | metadata_validation_split_0_by_camera | metadata_test_split_0_by_camera |
| S<sub>1</sub> | metadata_train_split_1_by_camera | metadata_validation_split_1_by_camera | metadata_test_split_1_by_camera |
| S<sub>2</sub> | metadata_train_split_2_by_camera | metadata_validation_split_2_by_camera | metadata_test_split_2_by_camera |
| S<sub>3</sub> | metadata_train_split_by_date | metadata_validation_split_by_date | metadata_test_split_by_date |
| S<sub>4</sub> | metadata_train_split_3_by_camera | metadata_validation_split_3_by_camera | metadata_test_split_3_by_camera |
| S<sub>5</sub> | metadata_train_split_4_by_camera | metadata_validation_split_4_by_camera | metadata_test_split_4_by_camera |

The following table shows the content in each split, except S<sub>3</sub>. The splitting strategy is that each view will be present in the testing set at least once. Also, the camera views that monitor different industrial facilities (view 1-0, 2-0, 2-1, and 2-2) are always on the testing set. Examples of the camera views will be provided later in this section.

| View | S<sub>0</sub> | S<sub>1</sub> | S<sub>2</sub> | S<sub>4</sub> | S<sub>5</sub> |
| --- | --- | --- | --- | --- | --- |
| 0-0 | Train | Train | Test | Train | Train |
| 0-1 | Test | Train | Train | Train | Train |
| 0-2 | Train | Test | Train | Train | Train |
| 0-3 | Train | Train | Validate | Train | Test |
| 0-4 | Validate | Train | Train | Test | Validate |
| 0-5 | Train | Validate | Train | Train | Test |
| 0-6 | Train | Train | Test | Train | Validate |
| 0-7 | Test | Train | Train | Validate | Train |
| 0-8 | Train | Train | Validate | Test | Train |
| 0-9 | Train | Test | Train | Validate | Train |
| 0-10 | Validate | Train | Train | Test | Train |
| 0-11 | Train | Validate | Train | Train | Test |
| 0-12 | Train | Train | Test | Train | Train |
| 0-13 | Test | Train | Train | Train | Train |
| 0-14 | Train | Test | Train | Train | Train |
| 1-0 | Test | Test | Test | Test | Test |
| 2-0 | Test | Test | Test | Test | Test |
| 2-1 | Test | Test | Test | Test | Test |
| 2-2 | Test | Test | Test | Test | Test |

The following shows the split of S<sub>3</sub> by time sequence, where the farthermost 18 days are used for training, the middle 2 days are used for validation, and the nearest 10 days are used for testing. You can find our camera data by date on [our air pollution monitoring network](http://mon.createlab.org/).
- Training set of S<sub>3</sub>
  - 2018-05-11, 2018-06-11, 2018-06-12, 2018-06-14, 2018-07-07, 2018-08-06, 2018-08-24, 2018-09-03, 2018-09-19, 2018-10-07, 2018-11-10, 2018-11-12, 2018-12-11, 2018-12-13, 2018-12-28, 2019-01-11, 2019-01-17, 2019-01-18
- Validation set of S<sub>3</sub>
  - 2019-01-22, 2019-02-02
- Testing set of S<sub>3</sub>
  - 2019-02-03, 2019-02-04, 2019-03-14, 2019-04-01, 2019-04-07, 2019-04-09, 2019-05-15, 2019-06-24, 2019-07-26, 2019-08-11

The dataset contains 12,567 clips with 19 distinct views from cameras on three sites that monitored three different industrial facilities. The clips are from 30 days that spans four seasons in two years in the daytime. The following provides examples and the distribution of labels for each camera view, with the format [camera_id]-[view_id]:

![This figure shows a part of the dataset.](back-end/data/dataset/2020-02-24/dataset_1.png)

![This figure shows a part of the dataset.](back-end/data/dataset/2020-02-24/dataset_2.png)

![This figure shows a part of the dataset.](back-end/data/dataset/2020-02-24/dataset_3.png)

![This figure shows a part of the dataset.](back-end/data/dataset/2020-02-24/dataset_4.png)

We made sure that we were not invading the privacy of surrounding residential neighbors. Areas in the videos that look inside house windows were cropped or blocked. Also, there is no law in our region to prohibit the monitoring of industrial activities.

# <a name="pretrained-models"></a>Pretrained models

We release two of our best baseline models: [RGB-I3D](back-end/data/pretrained_models/RGB-I3D-S3.pt) and [RGB-TC](https://github.com/CMU-CREATE-Lab/deep-smoke-machine/blob/master/back-end/data/pretrained_models/RGB-TC-S3.pt), both trained and tested on split S<sub>3</sub> using four NVIDIA GTX 1080 Ti GPUs. Please feel free to finetune your models based on our baseline. Our paper describes the details of these models. RGB-I3D uses [I3D ConvNet architecture with Inception-v1 layers](https://arxiv.org/pdf/1705.07750.pdf) and RGB frame input. RGB-TC is finetuned from RGB-I3D, with additional [Timeception](https://arxiv.org/pdf/1812.01289.pdf) layers. Below shows an example usage:
```python
# Import I3D
from i3d_learner import I3dLearner

# Initialize the model
model = I3dLearner(
    mode="rgb",
    augment=True,
    p_frame="../data/rgb/",
    use_tc=True,
    freeze_i3d=True,
    batch_size_train=8,
    milestones=[1000, 2000],
    num_steps_per_update=1)

# Finetune the RGB-TC model from the RGB-I3D model
model.fit(
    p_model="../data/pretrained_models/RGB-I3D-S3.pt",
    model_id_suffix="-s3",
    p_metadata_train="../data/split/metadata_train_split_by_date.json",
    p_metadata_validation="../data/split/metadata_validation_split_by_date.json",
    p_metadata_test="../data/split/metadata_test_split_by_date.json")
```

# <a name="deploy-models-to-recognize-smoke"></a>Deploy models to recognize smoke

We provide an example script ([recognize_smoke.py](back-end/www/recognize_smoke.py)) to show how you can deploy the trained models to recognize industrial smoke emissions. This script only works for the videos on our camera monitoring system ([http://mon.createlab.org/](http://mon.createlab.org/)) or others that are created using the [timemachine-creator](https://github.com/CMU-CREATE-Lab/timemachine-creator) and [timemachine-viewer](https://github.com/CMU-CREATE-Lab/timemachine-viewer). In sum, the script takes a list of video URLs (examples can be found [here](https://github.com/CMU-CREATE-Lab/deep-smoke-machine/blob/improve-documentation/back-end/data/production_url_list/2019-01-03.json)), gets their date and camera view boundary information, generates a bunch of cropped clips, and run the model on these clips to recognize smoke emissions. Here are the steps:

First, for a date that you want to process, create a JSON file under the [back-end/data/production_url_list/](back-end/data/production_url_list/) folder to add video URLs. The format of the file name must be "YYYY-MM-DD.json" (such as "2019-01-03.json"). If the file for that date exists, just open the file and add more video URLs. Each video URL is specified using a dictionary, and you need to put the video URLs in a list in each JSON file. For example:
```json
[{
  "url": "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2019-01-03.timemachine/&width=180&height=180&startFrame=9716&format=mp4&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=6304,884,6807,1387&nframes=36",
  "cam_id": 0,
  "view_id": 0
  },{
  "url": "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2019-01-03.timemachine/&width=180&height=180&startFrame=9716&format=mp4&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=6007,928,6509,1430&nframes=36",
  "cam_id": 0,
  "view_id": 1
}]
```

The URL indicates the cropped video clips, which is obtained by using the thumbnail tool on our camera monitoring system ([http://mon.createlab.org/](http://mon.createlab.org/)). To access the thumbnail tool, click the "share" button at the bottom-right near the timeline slider and then select the "Share as image or video" tab. A tutorial about how to use the thumbnail tool for sharing videos can be found [here](https://vimeo.com/140196813#t=415s). The cam_id and view_id correspond to the camera views presented in the "Dataset" section in this READEME. For example, if cam_id is 0 and view_id is 1, this means that the camera view is 0-1, as shown in [this graph](back-end/data/dataset/2020-02-24/dataset_1.png). After creating the JSON files or adding video URLs to existing JSON files, run the following to perform a sanity check, which will identify problems related to the camera data and attemp to fix the problems:
```sh
python recognize_smoke.py check_and_fix_urls
```

Next, run the following (this step will take a long time) to process each clip and predict the probability of having smoke:
```sh
python recognize_smoke.py process_all_urls
```

This will create a "production" folder under [back-end/data/](back-end/data) to store the processed results. Then, run the following to identify events based on the probabilities of having smoke:
```sh
python recognize_smoke.py process_events
```

This will create an "event" folder under [back-end/data/](back-end/data) to store the links to the video clips that are identified as having smoke emissions. To visualize the smoke events, copy the folder (with the same folder name, "event") to the front-end of the [video labeling tool](https://github.com/CMU-CREATE-Lab/video-labeling-tool/tree/master/front-end). Then, the [event page](https://smoke.createlab.org/event.html?date=2019-04-02&camera=0&view=all) will be able to access the "event" folder and show the results. You may also want to consider running the following to scan the video clips so that users do not need to wait for the [thumbnail server](https://thumbnails-v2.createlab.org/status) to render videos:
```sh
python recognize_smoke.py scan_urls
```

# <a name="admin"></a>Administrator only tasks

For researchers in our team, if you wish to update the dataset, you need to obtain user token from the [smoke labeling tool](https://smoke.createlab.org/gallery.html) and put the `user_token.js` file in the `deep-smoke-machine/back-end/data/` directory. You need permissions from the system administrator of the smoke labeling tool to download the user token. After getting the token, get the video metadata (using the command below). This will create a `metadata.json` file under `deep-smoke-machine/back-end/data/`.
```sh
# This is only for system
python get_metadata.py confirm
```

# <a name="acknowledgements"></a>Acknowledgements

We thank [GASP](https://gasp-pgh.org/) (Group Against Smog and Pollution), [Clean Air Council](https://cleanair.org/), [ACCAN](https://accan.org/) (Allegheny County Clean Air Now), [Breathe Project](https://breatheproject.org/), [NVIDIA](https://developer.nvidia.com/academic_gpu_seeding), and the [Heinz Endowments](http://www.heinz.org/) for the support of this research. We also greatly appreciate the help of our volunteers, which includes labeling videos and providing feedback in system development.
