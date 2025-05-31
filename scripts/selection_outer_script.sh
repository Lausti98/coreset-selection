#!/bin/bash
sbatch --job-name=SelTis --array 1-7 --cpus-per-task=4 --mem=6000M -p gpu --gres=gpu:titanrtx:1 --time=1-00:00:00 ./selection_script.sh