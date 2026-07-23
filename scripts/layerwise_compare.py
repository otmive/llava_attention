from plotter import Plotter
from functions import load_model
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration
from transformers import AutoModelForImageTextToText
import os 
import argparse
import multiprocessing as mp
import random
import pandas as pd

# answer_mapping = {
#     "positive": True,
#     "negative": True,
#     "positive_false": False,
#     "negative_false": False,
# }

# data_dir = "whatsup"
# model_name = 'paligemma'

# mentioned_attn_pos = []
# unmentioned_attn_pos = []
# mentioned_attn_neg = []
# unmentioned_attn_neg = []

# for image_file in os.listdir(os.path.join(data_dir, "images"))[0:20]:
#   for qtype in ['positive', 'negative', 'positive_false', 'negative_false']:
#         print("running for image ", image_file)
#         image_path = os.path.join(data_dir, "images", image_file)
#         print(f"Processing image: {image_path}")
#         plotter = Plotter(image_path)
#         left_colour = plotter.get_left_shapes()[0].colour
#         right_colour = plotter.get_right_shapes()[0].colour
#         processor, model = load_model(model_name)
#         plotter.set_model(model, processor)
#         # randomly decide whether to ask about the left or right object first to avoid biasing the model
#         pos1 = "left" if random.random() < 0.5 else "right"
#         if pos1 == "left":
#             col1, col2 = left_colour, right_colour
#         else:
#             col1, col2 = right_colour, left_colour

#         article = "a "
#         prompt_templates = {
#               "positive": f"Is this statement correct? The object on the {pos1} is {article}{col1}. Answer yes or no.",           # True: "Top left is Blue"
#               "negative": f"Is this statement correct? The object on the {pos1} is not {article}{col2}. Answer yes or no.",       # True: "Top left is not Red"
#               "positive_false": f"Is this statement correct? The object on the {pos1} is {article}{col2}. Answer yes or no.",     # False: "Top left is Red"
#               "negative_false": f"Is this statement correct? The object on the {pos1} is not {article}{col1}. Answer yes or no."  # False: "Top left is not Blue"
#         }
#         question = prompt_templates[qtype]
#         plotter.get_outputs(question)
#         model_answer = plotter.print_output()
#         print("Model answer:")
#         print(model_answer)
#         model_answer_bool = model_answer.lower().strip() in ["yes", "true", "correct"]
#         is_correct = model_answer_bool == answer_mapping[qtype]

#         bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()

#         if not is_correct:
#           if qtype == 'positive' or qtype == 'negative_false':
#             mentioned_colour = col1
#             unmentioned_colour = col2
#           else:
#             mentioned_colour = col2
#             unmentioned_colour = col1
#           if qtype == 'positive' or qtype == 'positive_false':
#             mentioned_attn_pos.append(bbox_attentions[mentioned_colour])
#             unmentioned_attn_pos.append(bbox_attentions[unmentioned_colour])
#           else:
#             mentioned_attn_neg.append(bbox_attentions[mentioned_colour])
#             unmentioned_attn_neg.append(bbox_attentions[unmentioned_colour])

# save_dir = "compare_save"
# image_dir = "whatsup"
# np.save(save_dir + f"/{model_name}_{image_dir}_wrong_mentioned_attn.npy", np.array(mentioned_attn_pos))
# np.save(save_dir + f"/{model_name}_{image_dir}_wrong_non_mentioned_attn.npy", np.array(unmentioned_attn_pos))

# np.save(save_dir + f"/{model_name}_{image_dir}_wrong_neg_mentioned_attn.npy", np.array(mentioned_attn_neg))
# np.save(save_dir + f"/{model_name}_{image_dir}_wrong_neg_non_mentioned_attn.npy", np.array(unmentioned_attn_neg))

def generate_layerwise(save_dir):
        fig, axs = plt.subplots(2, 2, figsize=(21, 8))

        # add title to entire figure
        fig.suptitle(f"PaliGemma", fontsize=20, fontweight='bold')#, y=1.02
        plt.style.use('tableau-colorblind10') # Good for accessibility
        plt.rcParams['font.family'] = 'sans-serif'
        models = ["paligemma", "paligemma_wrong"]

        # Modern color palette
        color_m = '#2c7fb8'  # Focused Blue
        color_nm = '#f03b20' # Focused Red/Orange
        color_base = '#636363' # Neutral Grey

        data_type = "whatsup"

        for i, model_n in enumerate(models):
            print(model_n)
            max_val = 0
            wrong_val = ""
            if model_n == 'paligemma_wrong':
              model_name = 'paligemma'
              wrong_val = "wrong_"
            else:
              model_name = "paligemma"
            print(model_name)
            for row, condition in enumerate(["pos", "neg"]):

                pretty_name = f"{model_name.capitalize()} - {'Affirmative' if condition == 'pos' else 'Negated'} " 
                neg_val = "neg_" if condition == "neg" else ""
                mentioned_attn = np.load(f"{save_dir}/{model_name}_{data_type}_{wrong_val}{neg_val}mentioned_attn.npy")
                not_mentioned_attn = np.load(f"{save_dir}/{model_name}_{data_type}_{wrong_val}{neg_val}non_mentioned_attn.npy")
                print(len(mentioned_attn))
                print(len(not_mentioned_attn))
                if len(mentioned_attn)>1:
  
                  if data_type == "multary":
                      avg_m = np.mean(mentioned_attn, axis=0)
                      print(f"{save_dir}/{model_name}_{data_type}_{wrong_val}{neg_val}mentioned_attn.npy")
                      print(avg_m.shape)
                      avg_nm = np.mean(not_mentioned_attn, axis=(0,1))
                      ci_m = np.std(mentioned_attn, axis=(0)) / np.sqrt(len(mentioned_attn)) * 1.96
                      ci_nm = np.std(not_mentioned_attn, axis=(0,1)) / np.sqrt(len(not_mentioned_attn)) * 1.96
                  else:   
                  
                      avg_m = np.mean(mentioned_attn, axis=0)
                      print(condition)
                      print(model_name)
                      print(f"{save_dir}/{model_name}_{data_type}_{wrong_val}{neg_val}mentioned_attn.npy")
                      print(avg_m.shape)

                      avg_nm = np.mean(not_mentioned_attn, axis=0)
                      print(f"{save_dir}/{model_name}_{data_type}_{wrong_val}{neg_val}non_mentioned_attn.npy")
                      print(avg_nm.shape)
                      ci_m = np.std(mentioned_attn, axis=0) / np.sqrt(len(mentioned_attn)) * 1.96
                      ci_nm = np.std(not_mentioned_attn, axis=0) / np.sqrt(len(not_mentioned_attn)) * 1.96
                  ax = axs[row, i]
                  x = list(range(len(avg_m)))
                  
                  max_val = max(np.max(avg_m + ci_m), np.max(avg_nm + ci_nm), max_val)
                  # --- Plotting ---
                  # # Baseline first so it stays in the background
                  # if plot_baseline:
                  #     ax.plot(x, baseline, label='Baseline', color=color_base, linestyle='--', alpha=0.8, linewidth=1.5)
                  
                  # Not Mentioned
                  ax.plot(x, avg_nm, label='Alternative', color=color_nm, linewidth=2.5, zorder=2)
                  ax.fill_between(x, avg_nm - ci_nm, avg_nm + ci_nm, color=color_nm, alpha=0.15)
                  
                  # Mentioned (Z-order higher to stand out)
                  ax.plot(x, avg_m, label='Mentioned', color=color_m, linewidth=2.5, zorder=3)
                  ax.fill_between(x, avg_m - ci_m, avg_m + ci_m, color=color_m, alpha=0.15)

                  # --- Formatting ---
                  # Titles only on the top row
                  ax.set_title(pretty_name, fontsize=16, fontweight='bold', pad=10)
                  
                  # Row labels (Y-axis labels)
                  ax.set_ylabel("Attention", fontsize=16, fontweight='normal')

                  wrong_title = "Correct" if wrong_val != "wrong_" else "Incorrect"
                  if condition == "neg":
                    wrong_title += " Negated" 
                  ax.set_title(wrong_title)
                  # Clean up Spines (The "Despine" look)
                  ax.spines['top'].set_visible(False)
                  ax.spines['right'].set_visible(False)
                  ax.grid(True, axis='y', linestyle=':', alpha=0.7)
                  
                  # Limits and Ticks
                  # Add some headroom above the max value
                  ax.tick_params(axis='both', which='major', labelsize=16)
                  
                  # Legend only on the first plot to avoid clutter
                  if i == 0 and row == 0:
                      ax.legend(frameon=False, fontsize=16, loc='upper left')#
            # set global y-limits based on max value across both conditions for this model
            axs[0, i].set_ylim(0, max_val * 1.1)
            axs[1, i].set_ylim(0, max_val * 1.1)

        # Global X-label
        fig.text(0.5, 0.01, 'Layer', ha='center', fontsize=16)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for global x-label
        plt.savefig(f"plots/{data_type}.png", dpi=300)

generate_layerwise("compare_save")
