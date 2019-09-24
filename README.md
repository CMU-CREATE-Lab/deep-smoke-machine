# deep-smoke-machine
Deep learning for smoke detection. The videos are from the [smoke labeling tool](https://github.com/CMU-CREATE-Lab/video-labeling-tool). The code in this repository assumes that Ubuntu 18.04 server is installed.

# Install Nvidia drivers, cuda, and cuDNN
Disable the nouveau driver.
```sh
sudo vim /etc/modprobe.d/blacklist.conf
# Add the following to this file
# Blacklist nouveau driver (for nvidia driver installation)
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```
Regenerate the kernel initramfs.
```sh
sudo update-initramfs -u
sudo reboot now
```
Remove old nvidia drivers.
```
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get autoremove
```
If using a desktop version of Ubuntu (not the server version), run the following:
```
sudo apt-get install ubuntu-desktop # only for desktop version, not server version
```
Install cuda and the nvidia driver. Documentation can be found on [Nvidia's website](https://docs.nvidia.com/cuda/).
```sh
sudo apt install build-essential
sudo apt-get install linux-headers-$(uname -r)
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_418.67_linux.run
sudo sh cuda_10.1.168_418.67_linux.run
```
Check if Nvidia driver is installed. Should be no nouveau.
```sh
sudo nvidia-smi
dpkg -l | grep -i nvidia
lsmod | grep -i nvidia
lspci | grep -i nvidia
lsmod | grep -i nouveau
dpkg -l | grep -i nouveau
```
Add cuda runtime library.
```sh
sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig
```
Add cuda environment path.
```sh
sudo vim /etc/environment
# add :/usr/local/cuda/bin (including the ":") at the end of the PATH="/[some_path]:/[some_path]" string (inside the quotes)
sudo reboot now
```
Check cuda installation.
```sh
cd /usr/local/cuda/samples
sudo make
cd /usr/local/cuda/samples/bin/x86_64/linux/release
./deviceQuery
```
Install cuDNN. Documentation can be found on [Nvidia's website](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux). Visit [Nvidia's page](https://developer.nvidia.com/cudnn) to download cuDNN to your local machine. Then, move the file to the Ubuntu server.
```sh
rsync -av /[path_on_local]/cudnn-10.1-linux-x64-v7.6.0.64.tgz [user_name]@[server_name]:[path_on_server]
ssh [user_name]@[server_name]
cd [path_on_server]
sudo tar -xzvf cudnn-10.1-linux-x64-v7.6.0.64.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
 
# Setup this tool
Clone this repository and set the permission.
```sh
git clone --recursive https://github.com/CMU-CREATE-Lab/deep-smoke-machine.git
sudo chown -R $USER deep-smoke-machine/
sudo addgroup [group_name]
sudo usermod -a -G [group_name] [user_name]
groups [user_name]
sudo chmod -R 775 deep-smoke-machine/
sudo chgrp -R [group_name] deep-smoke-machine/
```
For git to ignore permission changes.
```sh
# For only this repository
git config core.fileMode false

# For globally
git config --global core.fileMode false
```
Create conda environment and install packages. It is important to install pip first inside the newly created conda environment.
```sh
conda create -n deep-smoke-machine
conda activate deep-smoke-machine
conda install pip
which pip # make sure this is the pip inside the deep-smoke-machine environment
sh deep-smoke-machine/back-end/install_packages.sh
```
Update the optical_flow submodule.
```sh
cd deep-smoke-machine/back-end/www/optical_flow/
git submodule update --init --recursive
git checkout master
```
Install PyTorch.
```sh
# For cuda 9.0
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# For cuda 10.0
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
Install system packages for OpenCV.
```sh
sudo apt update
sudo apt install -y libsm6 libxext6 libxrender-dev
```

# Using Tensorboard
Create a logging directory
```
mkdir run
```
Write data to the model_runs directory while running the model. After writing model data to the directory, launch tensorboard.
```
tensorboard --logdir=run
```
After launching, tensorboard will start a server. To view, navigate to the stated URL in your browser. For more information about data input types, refer to [the official documentation](https://pytorch.org/docs/stable/tensorboard.html)

# Use this tool
Obtain user token from the [smoke labeling tool](https://smoke.createlab.org/gallery.html) and put the user_token.js file in the deep-smoke-machine/back-end/data directory. You need permissions from the system administrator to download the user token. After getting the token, get the video metadata.
```sh
python get_metadata.py confirm
```
Split the metadata into three sets: train, validation, and test.
```sh
python split_metadata.py confirm
```
Download all videos in the metadata file.
```sh
python download_videos.py

# Background script (on the background using the "screen" command)
sh bg.sh python download_videos.py
```
Process all videos into rgb frames and optical flows, then save all data to the disk.
```sh
python process_videos.py

# Background script (on the background using the "screen" command)
sh bg.sh python process_videos.py
```
Extract [I3D features](https://github.com/piergiaj/pytorch-i3d).
```sh
python extract_features.py [method] [optional_model_path]

# Extract features from pretrained i3d
python extract_features.py i3d-rgb
python extract_features.py i3d-flow

# Extract features from a saved i3d model
python extract_features.py i3d-rgb ../data/saved_i3d/ecf7308-i3d-rgb/model/16875.pt
python extract_features.py i3d-flow ../data/saved_i3d/af00751-i3d-flow/model/30060.pt

# Background script (on the background using the "screen" command)
sh bg.sh python extract_features.py i3d-rgb
sh bg.sh python extract_features.py i3d-flow
```
Train the model with the training and validation sets. Pretrained weights are obtained from the [pytorch-i3d repository](https://github.com/piergiaj/pytorch-i3d).
- [Two-Stream Inflated 3D ConvNet](https://arxiv.org/abs/1705.07750)
- [Two-Stream ConvNet](http://papers.nips.cc/paper/5353-two-stream-convolutional)
```sh
python train.py [method] [optional_model_path]

# Use I3D features + SVM (rgb or flow mode)
python train.py svm-rgb
python train.py svm-flow

# Use Two-Stream Inflated 3D ConvNet (rgb or flow mode)
python train.py i3d-rgb
python train.py i3d-flow

# Background script (on the background using the "screen" command)
sh bg.sh python train.py i3d-rgb
sh bg.sh python train.py i3d-flow

# Use Two-Stream Inflated 3D ConvNet and resume from a saved model (rgb or flow mode)
python train.py i3d-rgb ../data/saved_i3d/ecf7308-i3d-rgb/model/16875.pt
python train.py i3d-flow ../data/saved_i3d/af00751-i3d-flow/model/30060.pt

# Use Two-Stream ConvNet (rgb or flow mode)
python train.py ts-rgb
python train.py ts-flow
```
Test the performance of a model on the test set.
```sh
python test.py [method] [model_path]

# Use I3D features + SVM (rgb or flow mode)
python test.py svm-rgb ../data/saved_svm/445cc62-svm-rgb/model/model.pkl
python test.py svm-flow ../data/saved_svm/b376702-svm-flow/model/model.pkl

# Use Two-Stream Inflated 3D ConvNet (rgb or flow mode)
python test.py i3d-rgb ../data/saved_i3d/ecf7308-i3d-rgb/model/16875.pt
python test.py i3d-flow ../data/saved_i3d/af00751-i3d-flow/model/30060.pt
```
Recommended training strategy:
1. Set an initial learning rate (e.g., 0.001)
2. Keep this learning rate and train the model until the training error decreases too slow (or fluctuate) or until the validation error increases (a sign of overfitting)
3. Decrease the learning rate (e.g., by a factor of 10)
4. Load the best model weight from the ones that were trained using the previous learning rate
5. Repeat step 2, 3, and 4 until convergence
