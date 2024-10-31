#!/bin/bash
#SBATCH --job-name="dgf_0_45_1_1_01"
#SBATCH --output=dgf_0_45_1_1_01.out
#SBATCH --error=err.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
module purge
module load Anaconda3
cd ..
source domaindensity/bin/activate
cd celeba
python -u sisa_4.py --test_domain 'ind_black' --seed 45 --fair_weight 1 --alpha 0.1 