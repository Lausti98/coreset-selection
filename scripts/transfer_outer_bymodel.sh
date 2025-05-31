#!/bin/bash
sbatch --job-name=TrEffi \
        --array 4-10%3 --cpus-per-task=4 \
        --mem=6000M -p gpu \
        --gres=gpu:titanrtx:1 \
        --time=1-00:00:00 \
        ./transfer_script_bymodel.sh