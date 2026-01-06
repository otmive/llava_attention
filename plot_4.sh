#!/bin/bash
#SBATCH --job-name=4_shapes_positive
#SBATCH --time=48:00:00
#SBATCH --mem=32G 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python gen_4_object_graphs.py