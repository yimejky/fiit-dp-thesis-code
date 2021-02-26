#!/bin/bash

MemoryRate=1000 # max 2200
ClockRate=200
PowerLimit=200 # watts


sudo nvidia-smi -pm 1
sudo nvidia-settings -a "[gpu:0]/GPUPowerMizerMode=1"
sudo nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffset[3]=$MemoryRate"
sudo nvidia-settings -a "[gpu:0]/GPUGraphicsClockOffset[3]=$ClockRate"
sudo nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffset[4]=$MemoryRate"
sudo nvidia-settings -a "[gpu:0]/GPUGraphicsClockOffset[4]=$ClockRate"
sudo nvidia-smi -pl $PowerLimit