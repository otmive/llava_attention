#!/bin/bash
#SBATCH --job-name=2_obj
#SBATCH --time=72:00:00
#SBATCH --mem=32G 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

nvidia-smi
# python gen_2_object_graphs.py --model llava --image_folder whatsup --ylim 0.3 --save_folder plots/whatsup_total_llava_plots
# python gen_2_object_graphs.py --model internvl --image_folder whatsup --ylim 0.4 --save_folder plots/whatsup_total_llava_plots
# python gen_2_object_graphs.py --model paligemma --image_folder whatsup --ylim 0.4 --save_folder plots/whatsup_total_llava_plots

python gen_2_and_4_plots.py
#python downstream_tasks.py