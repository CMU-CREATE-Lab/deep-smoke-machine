# Delete existing screen
for session in $(sudo screen -ls | grep -o '[0-9]*.run_tensorboard_ts')
do
  sudo screen -S "${session}" -X quit
  sleep 2
done

sudo rm tb.0

# For python in conda env
sudo screen -dmSL "run_tensorboard_ts" -Logfile tb.0 bash -c "export PATH='/opt/miniconda3/bin:$PATH'; . '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate deep-smoke-machine; tensorboard --logdir=../data/ts_runs"

# List screens
sudo screen -ls
