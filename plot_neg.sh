#!/bin/bash
#SBATCH --job-name=2_all
#SBATCH --time=48:00:00
#SBATCH --mem=32G 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#python gen_2_object_graphs_grouped.py
python gen_2_object_graphs.py