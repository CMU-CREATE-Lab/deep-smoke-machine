#!/bin/bash

# Before using this script, do:
#   sudo apt install -y xorg
#   sudo nvidia-xconfig --enable-all-gpus --allow-empty-initial-configuration --cool-bits=4
# A xorg config file will be created at:
#   cat /etc/X11/xorg.conf

sudo killall Xorg
sleep 2

# Checks
DEFAULT_DELAY=0
if [ "x$1" = "x" -o "x$1" = "xnone" ]; then
 DELAY=$DEFAULT_DELAY
else
 DELAY=$1
fi
sleep $DELAY

# Write log file for debugging
exec 2> /tmp/gpu_fan_settings.log
exec 1>&2
set -x

# Make sure an Xorg server is running which you can connect to
# ":1" can be any unused display number.
sudo systemctl set-default graphical.target
sudo X :1 &
sleep 10
sudo export DISPLAY=:1

# set GPU 0..N
sudo env DISPLAY=:1 nvidia-settings -a [gpu:0]/GPUFanControlState=1 -a [fan-0]/GPUTargetFanSpeed=60
sudo env DISPLAY=:1 nvidia-settings -a [gpu:1]/GPUFanControlState=1 -a [fan-1]/GPUTargetFanSpeed=60
sudo env DISPLAY=:1 nvidia-settings -a [gpu:2]/GPUFanControlState=1 -a [fan-2]/GPUTargetFanSpeed=60
sudo env DISPLAY=:1 nvidia-settings -a [gpu:3]/GPUFanControlState=1 -a [fan-3]/GPUTargetFanSpeed=60

# set back to command line prompt after boot (optional)
sudo systemctl set-default multi-user.target

# terminating Xorg will reset fan speeds to default. Terminate xorg as follows:
# sudo killall Xorg
