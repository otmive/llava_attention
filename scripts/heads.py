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

def run_mult_head_diff(prompt_type):

  if prompt_type == "positive" or prompt_type == "negative":
    target = "yes"
  else:
    target = "no"

  count = 0
  diffs = np.zeros((26))

  correct_top_left = {
        "positive": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "negative": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10],       # True: "Top left is not Red"
        "positive_false": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],     # False: "Top left is Red"
        "negative_false": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # False: "Top left is not Blue"
    }
  
  correct_top_right = {
        "positive": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "negative": [0, 1, 5, 9, 10, 11, 12, 14, 15, 16],       # True: "Top left is not Red"
        "positive_false": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],     # False: "Top left is Red"
        "negative_false": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # False: "Top left is not Blue"
    }

  correct_bottom_left = {
      "positive": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      "negative": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11],       # True: "Top left is not Red"
      "positive_false": [3, 5, 6, 7, 8, 9, 12, 14, 16, 17],     # False: "Top left is Red"
      "negative_false": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # False: "Top left is not Blue"
  }

  correct_bottom_right = {
      "positive": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      "negative": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],       # True: "Top left is not Red"
      "positive_false": [0, 1, 2, 3, 4, 5, 7, 9, 10, 11],     # False: "Top left is Red"
      "negative_false": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # False: "Top left is not Blue"
  }

  all_position_data = {
      "top_left": correct_top_left,
      "top_right": correct_top_right,
      "bottom_left": correct_bottom_left,
      "bottom_right": correct_bottom_right
  }

  processor, model = load_model("paligemma")

  # 2. Outer loop: Iterate through each position and its data dictionary
  for position, position_dict in all_position_data.items():
      print(f"Processing position: {position}")
      
      # Get the specific list of indices for the current prompt_type
      image_indices = position_dict[prompt_type]
      
      # Inner loop: Run the index loop for this specific position
      for i in image_indices:
          image_path = f"multary/images/image_{i:04d}.png"
          print(f"Processing image: {image_path}")
          plotter = Plotter(image_path)
          plotter = Plotter(image_path)
          top_left_colour = plotter.get_shape_by_position('top_left').colour
          top_right_colour = plotter.get_shape_by_position('top_right').colour
          bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
          bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour
          plotter.set_model(model, processor)

          if position == "top_left":
            pos1 = "top left"
            col1 = top_left_colour
          elif position == "top_right":
            pos1 = "top right"
            col1 = top_right_colour
          elif position == "bottom_right":
            pos1 = "bottom right"
            col1 = bottom_right_colour
          elif position == "bottom_left":
            pos1 = "bottom left"
            col1 = bottom_left_colour

          colours = ["red", "green", "blue", "yellow"]
          remaining_colours = [c for c in colours if c.lower() != col1.lower()]
          col2 = remaining_colours[0]

          prompts = {
                  "positive": f"Is the object on the {pos1} {col1}? Answer yes or no.",           # True: "Top left is Blue"
                  "negative": f"Is the object on the {pos1} not {col2}? Answer yes or no.",       # True: "Top left is not Red"
                  "positive_false": f"Is the object on the {pos1} {col2}? Answer yes or no.",     # False: "Top left is Red"
                  "negative_false": f"Is the object on the {pos1} not {col1}? Answer yes or no."  # False: "Top left is not Blue"  # False: "Top left is not Blue"
          }


          prob = plotter.get_outputs_prob(prompts[prompt_type], target)
          print("Orig Prob: ", prob)

   
          for l in range(26):
              altered_prob = plotter.get_outputs_prob(prompts[prompt_type], target, l)
              diff = prob - altered_prob
              diffs[l] += diff
          count +=1


  avg_diffs = diffs / 40
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

  print(avg_diffs)
  print(count)

  np.save(f"{prompt_type}_layers_multary.npy", avg_diffs)

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

def find_correct_mult(prompt_type, position):

  if prompt_type == "positive" or prompt_type == "negative":
    target = "yes"
  else:
    target = "no"


  correct = []
  processor, model = load_model("paligemma")
  for i in range(35):
    if len(correct) >= 10:
      break
    image_path = f"multary/images/image_{i:04d}.png"
    print(f"Processing image: {image_path}")

    plotter = Plotter(image_path)
    top_left_colour = plotter.get_shape_by_position('top_left').colour
    top_right_colour = plotter.get_shape_by_position('top_right').colour
    bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
    bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour
    plotter.set_model(model, processor)

    if position == "top_left":
      pos1 = "top left"
      col1 = top_left_colour
    elif position == "top_right":
      pos1 = "top right"
      col1 = top_right_colour
    elif position == "bottom_right":
      pos1 = "bottom right"
      col1 = bottom_right_colour
    elif position == "bottom_left":
      pos1 = "bottom left"
      col1 = bottom_left_colour

    colours = ["red", "green", "blue", "yellow"]
    remaining_colours = [c for c in colours if c.lower() != col1.lower()]
    col2 = remaining_colours[0]

    prompts = {
            "positive": f"Is the object on the {pos1} {col1}? Answer yes or no.",           # True: "Top left is Blue"
            "negative": f"Is the object on the {pos1} not {col2}? Answer yes or no.",       # True: "Top left is not Red"
            "positive_false": f"Is the object on the {pos1} {col2}? Answer yes or no.",     # False: "Top left is Red"
            "negative_false": f"Is the object on the {pos1} not {col1}? Answer yes or no."  # False: "Top left is not Blue"  # False: "Top left is not Blue"
    }

    plotter.get_outputs(prompts[prompt_type])
    out = plotter.print_output()
    print(prompts[prompt_type])
    print(out)
    if out == target:
      if len(correct) < 10:
        correct.append(i)
  print(correct)
  print(len(correct))

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

#run_head_diff("multary", "positive")
#find_correct_mult("negative_false", "bottom_right")
#run_mult_head_diff("negative_false")
#print_key_heads("positive")

pos = np.load("positive_layers_binary.npy")
neg = np.load("negative_layers_binary.npy")
pos_false = np.load("positive_false_layers_binary.npy")
neg_false = np.load("negative_false_layers_binary.npy")

pos_diffs = (neg + neg_false) / 2

fig, ax = plt.subplots(figsize=(4, 3), dpi=300) 

bar_color = '#2e6f77'
text_color = '#2c3e50'  


bars = ax.bar(range(26), pos_diffs, color=bar_color, width=0.7, edgecolor=bar_color, linewidth=0.2, zorder=3)


ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='#cccccc', zorder=0)
ax.grid(False, axis='x') 

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
    
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')


ax.set_title("Binary - Negated", fontsize=10, pad=8, fontweight='bold', color=text_color)
ax.set_xlabel("Layer", fontsize=9, labelpad=4, color=text_color)
ax.set_ylabel("Avg. Probability Difference", fontsize=9, labelpad=4, color=text_color)


ax.set_xticks(range(0, 26, 5))
ax.set_xticklabels(range(0, 26, 5), fontsize=8, color=text_color)
ax.tick_params(axis='y', labelsize=8, colors=text_color)


ax.set_xlim(-1, 26)


plt.tight_layout()
plt.savefig("neg_binary.png", bbox_inches='tight', dpi=300)
plt.show()