#!/bin/bash
sbatch --job-name=SelClu --array 1-5 --cpus-per-task=4 --mem=6000M -p gpu --gres=gpu:titanrtx:1 --time=1-00:00:00 ./selection_script.sh