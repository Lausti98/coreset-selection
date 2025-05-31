#!/bin/bash
sbatch --job-name=TrFull \
        --array 3-7%2 --cpus-per-task=4 \
        --mem=6000M -p gpu \
        --gres=gpu:titanrtx:1 \
        --time=1-00:00:00 \
        ./training_full_script.sh