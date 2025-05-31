#!/bin/bash
sbatch --job-name=TrOrg \
        --array 1-6%3 --cpus-per-task=4 \
        --mem=6000M -p gpu \
        --gres=gpu:titanrtx:1 \
        --time=1-00:00:00 \
        ./training_script_bymethod.sh