#!/bin/bash
#SBATCH --job-name=4_obj_graphs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --account=cosc030084
#SBATCH --partition gpu
#SBATCH --gres=gpu:2

python gen_4_object_graphs.py
