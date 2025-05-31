#!/bin/bash
sbatch --job-name=TrSmall \
        --array 1-37%2 --cpus-per-task=4 \
        --mem=6000M -p gpu \
        --gres=gpu:titanrtx:1 \
        --time=04:00:00 \
        ./training_1percent_script.sh
