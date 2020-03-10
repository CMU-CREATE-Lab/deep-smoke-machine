# Delete existing screen
for session in $(sudo screen -ls | grep -o '[0-9]*.train_fuse')
do
  sudo screen -S "${session}" -X quit
  sleep 2
done

# Delete the log
sudo rm screenlog.0

# For python in conda env
sudo screen -dmSL "train_fuse" bash -c "export PATH='/opt/miniconda3/bin:$PATH'; . '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate deep-smoke-machine; python train.py fuse"

# List screens
sudo screen -ls
