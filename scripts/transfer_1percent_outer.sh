#!/bin/bash
sbatch --job-name=TransferDense \
        --array 1-10%2 --cpus-per-task=4 \
        --mem=6000M -p gpu \
        --gres=gpu:titanrtx:1 \
        --time=08:00:00 \
        ./transfer_1percent_script.sh
