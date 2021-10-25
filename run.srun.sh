#!/bin/bash

#SBATCH -J TEP
#SBATCH -t 48:00:00
#SBATCH -N 1 -n 4
#SBATCH --mem 16GB
##SBATCH -p gpu
#SBATCH --gres=gpu:1

#SBATCH -A laszka

#SBATCH --array=1-5

##module load GCC/7.2.0-2.29
##module load Anaconda3/python-3.6
###module load cuDNN/7.5.0-CUDA-10.0.130

source /project/cacds/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate tep-gpu

cd /home/teghtesa/TennesseeEastmanProcess
export PATH=$PWD/gambit-project/:$PATH

python safety_test.py $SLURM_ARRAY_TASK_ID
cp -r $TMPDIR/tb_logs /home/teghtesa/TennesseeEastmanProcess/new_logs