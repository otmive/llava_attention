#!/bin/bash
#SBATCH --job-name=4_shapes_positive
#SBATCH --time=8:00:00
#SBATCH --mem=32G 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python positive_negated_diff_4.py