#!/bin/bash
#SBATCH --job-name=2_shapes
#SBATCH --time=8:00:00
#SBATCH --mem=32G 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

nvidia-smi
python whatsup.py
