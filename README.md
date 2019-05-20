# deep-smoke-machine
Deep learning for smoke detection.
```sh
git clone https://github.com/CMU-CREATE-Lab/deep-smoke-machine.git
sudo chown -R $USER deep-smoke-machine
conda create -n deep-smoke-machine
conda activate deep-smoke-machine
conda install pip
which pip # make sure this is the pip inside the deep-smoke-machine environment
sh deep-smoke-machine/back-end/install_packages.sh
cd deep-smoke-machine/back-end/www
```
Get video metadata from the [smoke labeling tool](https://github.com/CMU-CREATE-Lab/video-labeling-tool).
```sh
python get_metadata.py
```
Download all videos in the metadata file.
```sh
python download_videos.py
```
Extract [I3D features](https://github.com/piergiaj/pytorch-i3d).
```sh
python extract_features.py
```
Split the metadata into three sets: train, validation, and test.
```sh
python split_metadata.py
```
Fit and validate the model.
- [Two-Stream Inflated 3D ConvNet](https://arxiv.org/abs/1705.07750)
- [Two-Stream ConvNet](http://papers.nips.cc/paper/5353-two-stream-convolutional)
```sh
# Use I3D features + SVM
python validate.py feature

# Use Two-Stream Inflated 3D ConvNet
python validate.py i3d

# Use Two-Stream ConvNet
python validate.py ts
```
