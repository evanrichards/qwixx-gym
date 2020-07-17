#!/usr/bin/env bash
tar --exclude='./venv' \
 --exclude='./qwixx_gym.egg-info' \
 --exclude='./wandb' \
 --exclude='./.idea' \
 --exclude='./dqn_qwixx-v0_weights.h5f' \
 --exclude='./__pycache__' \
 -zcvf ~/Desktop/qwixx.tar.gz .