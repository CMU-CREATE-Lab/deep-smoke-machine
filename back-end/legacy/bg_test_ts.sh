# Delete existing screen
for session in $(sudo screen -ls | grep -o '[0-9]*.test_ts')
do
  sudo screen -S "${session}" -X quit
  sleep 2
done

# Delete the log
sudo rm screenlog.0

# For python in conda env
sudo screen -dmSL "test_ts" bash -c "export PATH='/opt/miniconda3/bin:$PATH'; . '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate deep-smoke-machine; python test.py ts ../data/saved_ts/07-16-19/0e1a321-ts-rgb/18900.pt"

# List screens
sudo screen -ls
