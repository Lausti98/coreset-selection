#!/bin/bash
module load python/3.12.8
pip3 install -r requirements.txt
cd src/
export PYTHONPATH=$PYTHONPATH:/home/vbp790/proj/
echo ${PYTHONPATH}
python3 evaluation/coreset_selector_executor.py 6 ${SLURM_ARRAY_TASK_ID}