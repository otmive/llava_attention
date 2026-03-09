#!/bin/bash
#SBATCH --job-name=2_shapes
#SBATCH --time=8:00:00
#SBATCH --mem=32G 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

nvidia-smi
# python gen_2_object_graphs.py --model llava --image_folder 2d_dataset_left_large_1000 --ylim 0.015 --save_folder plots/left_large_plots
# python gen_2_object_graphs.py --model llava --image_folder 2d_dataset_right_large_1000 --ylim 0.015 --save_folder plots/right_large_plots
#python downstream_tasks.py
python downstream.py