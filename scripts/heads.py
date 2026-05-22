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
import numpy as np
import seaborn as sns

def run_head_diff(dataset, prompt_type):

  if prompt_type == "positive" or prompt_type == "negative":
    target = "yes"
  else:
    target = "no"

  diffs = np.zeros((26))

  if dataset == "binary":
    correct_left = {
          "positive": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
          "negative": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],       # True: "Top left is not Red"
          "positive_false": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],     # False: "Top left is Red"
          "negative_false": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] # False: "Top left is not Blue"
      }
    
    correct_right = {
          "positive": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
          "negative": [2, 3, 4, 5, 7, 8, 9, 11, 13, 14, 16, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34],       # True: "Top left is not Red"
          "positive_false": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],     # False: "Top left is Red"
          "negative_false": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # False: "Top left is not Blue"
      }

  processor, model = load_model("paligemma")
  for i in correct_left[prompt_type]:
    image_path = f"{dataset}/images/image_{i:04d}.png"
    print(f"Processing image: {image_path}")
    plotter = Plotter(image_path)
    left_colour = plotter.get_left_shapes()[0].colour
    right_colour = plotter.get_right_shapes()[0].colour
    plotter.set_model(model, processor)

    article = ""
    pos1 = "left"
    col1 = left_colour
    col2 = right_colour
    prompts = {
              "positive": f"Is this statement correct? The object on the {pos1} is {article}{col1}. Answer yes or no.",           # True: "Top left is Blue"
              "negative": f"Is this statement correct? The object on the {pos1} is not {article}{col2}. Answer yes or no.",       # True: "Top left is not Red"
              "positive_false": f"Is this statement correct? The object on the {pos1} is {article}{col2}. Answer yes or no.",     # False: "Top left is Red"
              "negative_false": f"Is this statement correct? The object on the {pos1} is not {article}{col1}. Answer yes or no."  # False: "Top left is not Blue"
          }

    prob = plotter.get_outputs_prob(prompts[prompt_type], target)
    print("Orig Prob: ", prob)

   
    for l in range(26):
        altered_prob = plotter.get_outputs_prob(prompts[prompt_type], target, l)
        diff = prob - altered_prob
        diffs[l] += diff

  for i in correct_right[prompt_type]:
    image_path = f"binary/images/image_{i:04d}.png"
    print(f"Processing image: {image_path}")
    plotter = Plotter(image_path)
    left_colour = plotter.get_left_shapes()[0].colour
    right_colour = plotter.get_right_shapes()[0].colour
    plotter.set_model(model, processor)

    article = ""
    pos1 = "right"
    col1 = right_colour
    col2 = left_colour
    prompts = {
                "positive": f"Is this statement correct? The object on the {pos1} is {article}{col1}. Answer yes or no.",           # True: "Top left is Blue"
                "negative": f"Is this statement correct? The object on the {pos1} is not {article}{col2}. Answer yes or no.",       # True: "Top left is not Red"
                "positive_false": f"Is this statement correct? The object on the {pos1} is {article}{col2}. Answer yes or no.",     # False: "Top left is Red"
                "negative_false": f"Is this statement correct? The object on the {pos1} is not {article}{col1}. Answer yes or no."  # False: "Top left is not Blue"
            }

    prob = plotter.get_outputs_prob(prompts[prompt_type], target)
    print("Orig Prob: ", prob)

   
    for l in range(26):
        altered_prob = plotter.get_outputs_prob(prompts[prompt_type], target, l)
        diff = prob - altered_prob
        diffs[l] += diff

  plt.figure(figsize=(10, 12))  # Tall figure because 26 layers > 8 heads

  avg_diffs = diffs / 50
  # sns.heatmap makes the grid. 'annot=False' hides individual numbers so it isn't cluttered,
  # but you can set 'annot=True' if you want to see the raw numbers in the cells.
  #ax = sns.heatmap(avg_diffs, cmap="viridis", annot=False, fmt=".4f", cbar_kws={'label': 'Difference'})

  # plt.title("Average Probability Difference", fontsize=20, pad=15)
  # plt.xlabel("Attention Head", fontsize=20)
  # plt.ylabel("Layer", fontsize=20)

  # # Adjust ticks to match 0-indexed layers and heads
  # plt.xticks(np.arange(8) + 0.5, labels=range(8))
  # plt.yticks(np.arange(26) + 0.5, labels=range(26), rotation=0)
  # ax.figure.axes[-1].yaxis.label.set_size(18) # Colorbar label
  # ax.figure.axes[-1].tick_params(labelsize=18)
  # ax.tick_params(axis='both', which='major', labelsize=18)
  # plt.tight_layout()
  # plt.savefig(f"{prompt_type}_layers.png")

  np.save(f"{prompt_type}_layers_{dataset}.npy", avg_diffs)

def find_correct(dataset, prompt_type):

  if prompt_type == "positive" or prompt_type == "negative":
    target = "yes"
  else:
    target = "no"

  left_correct = []
  right_correct = []
  processor, model = load_model("paligemma")
  for i in range(35):
    if len(left_correct) >= 25 and len(right_correct) >= 25:
      break
    image_path = f"{dataset}/images/image_{i:04d}.png"
    print(f"Processing image: {image_path}")


    for pos in ['left', 'right']:
      plotter = Plotter(image_path)
      left_colour = plotter.get_left_shapes()[0].colour
      right_colour = plotter.get_right_shapes()[0].colour
      plotter.set_model(model, processor)
      article = ""
      pos1 = pos
      if pos == "left":
        col1 = left_colour
        col2 = right_colour
      else:
        col1 = right_colour
        col2 = left_colour
      prompts = {
          "positive": f"Is this statement correct? The object on the {pos1} is {article}{col1}. Answer yes or no.",           # True: "Top left is Blue"
          "negative": f"Is this statement correct? The object on the {pos1} is not {article}{col2}. Answer yes or no.",       # True: "Top left is not Red"
          "positive_false": f"Is this statement correct? The object on the {pos1} is {article}{col2}. Answer yes or no.",     # False: "Top left is Red"
          "negative_false": f"Is this statement correct? The object on the {pos1} is not {article}{col1}. Answer yes or no."  # False: "Top left is not Blue"
      }

      plotter.get_outputs(prompts[prompt_type])
      out = plotter.print_output()
      print(prompts[prompt_type])
      print(out)
      if out == target:
        if pos == "left" and len(left_correct) < 25:
          left_correct.append(i)
        elif pos == "right" and len(right_correct) < 25:
          right_correct.append(i)

  print("correct images for left: ", left_correct)
  print("correct images for right ", right_correct)
  print(len(left_correct))
  print(len(right_correct))

def print_key_heads(prompt_type):
  data = np.load(f"{prompt_type}.npy")
  items = []
  for r in range(data.shape[0]):
      for c in range(data.shape[1]):
          items.append(((r, c), data[r, c]))

  # 2. Sort the list based on the value (item[1]) in descending order
  items.sort(key=lambda item: item[1], reverse=True)

  # 3. Print out the top 5
  print("--- Top 5 Highest Values ---")
  for rank in range(5):
      coords, value = items[rank]
      row, col = coords
      print(f"{rank+1}: Value {value:.4f} at Layer {row}, Head {col}")

#run_head_diff("binary", "negative_false")

#print_key_heads("positive")

pos = np.load("positive_layers_binary.npy")
neg = np.load("negative_layers_binary.npy")
pos_false = np.load("positive_false_layers_binary.npy")
neg_false = np.load("negative_false_layers_binary.npy")

pos_diffs = (neg + neg_false) / 2
plt.figure(figsize=(10, 6))
plt.bar(range(26), pos_diffs, color='steelblue')
plt.title("Average Probability Difference by Layer", fontsize=20, pad=15)
plt.xlabel("Layer", fontsize=20)
plt.ylabel("Average Probability Difference", fontsize=20)
plt.xticks(range(26), fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(f"neg_layers_tot.png")