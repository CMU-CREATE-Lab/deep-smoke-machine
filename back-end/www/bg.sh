#!/bin/sh

# This script runs a python script using screen command
# For example:
# sh bg.sh python download_videos.py
# sh bg.sh python process_videos.py

# Get file path
if [ "$1" != "" ] && [ "$2" != "" ]
then
  echo "Run: $1 $2 $3"
else
  echo "Usage: sh bg_python.sh python [script_path] [script_variable]"
  exit 1
fi

# Delete existing screen
for session in $(sudo screen -ls | grep -o "[0-9]*.$1.$2.$3")
do
  sudo screen -S "${session}" -X quit
  sleep 2
done

# Delete the log
sudo rm screenlog.0

# For python in conda env in Ubuntu
sudo screen -dmSL "$1.$2.$3" bash -c "export PATH='/opt/miniconda3/bin:$PATH'; . '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate deep-smoke-machine; $1 $2 $3"

# List screens
sudo screen -ls
exit 0
