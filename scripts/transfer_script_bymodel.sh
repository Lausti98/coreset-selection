#!/bin/bash

#SBATCH --ntasks=1 --cpus-per-task=4 --mem=6000M

#SBATCH -p gpu --gres=gpu:titanrtx:1

#SBATCH --time=4:00:00

module load python/3.12.8
pip3 install -r requirements.txt
cd src/
export PYTHONPATH=$PYTHONPATH:/home/vbp790/proj/
echo ${PYTHONPATH}
python3 evaluation/transfer_executor.py ${SLURM_ARRAY_TASK_ID} clusteringmosotoefficientnet 