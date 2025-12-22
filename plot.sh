#!/bin/bash
#SBATCH --job-name=2_shape
#SBATCH --time=8:00:00
#SBATCH --mem=32G 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python four_shapes.py