#!/bin/bash
#SBATCH --job-name=left_large
#SBATCH --time=8:00:00
#SBATCH --mem=32G 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python what_colour.py