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

answer_mapping = {
    "positive": True,
    "negative": True,
    "positive_false": False,
    "negative_false": False,
}

data_dir = "binary"
model_name = 'paligemma'

mentioned_attn_pos = []
unmentioned_attn_pos = []
mentioned_attn_neg = []
unmentioned_attn_neg = []

for image_file in os.listdir(os.path.join(data_dir, "images"))[0:1]:
  for qtype in ['positive', 'negative', 'positive_false', 'negative_false']:
        print("running for image ", image_file)
        image_path = os.path.join(data_dir, "images", image_file)
        print(f"Processing image: {image_path}")
        plotter = Plotter(image_path)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour
        processor, model = load_model(model_name)
        plotter.set_model(model, processor)
        # randomly decide whether to ask about the left or right object first to avoid biasing the model
        pos1 = "left" if random.random() < 0.5 else "right"
        if pos1 == "left":
            col1, col2 = left_colour, right_colour
        else:
            col1, col2 = right_colour, left_colour

        article = ""
        prompt_templates = {
              "positive": f"Is this statement correct? The object on the {pos1} is {article}{col1}. Answer yes or no.",           # True: "Top left is Blue"
              "negative": f"Is this statement correct? The object on the {pos1} is not {article}{col2}. Answer yes or no.",       # True: "Top left is not Red"
              "positive_false": f"Is this statement correct? The object on the {pos1} is {article}{col2}. Answer yes or no.",     # False: "Top left is Red"
              "negative_false": f"Is this statement correct? The object on the {pos1} is not {article}{col1}. Answer yes or no."  # False: "Top left is not Blue"
        }
        question = prompt_templates[qtype]
        plotter.get_outputs(question)
        model_answer = plotter.print_output()
        print("Model answer:")
        print(model_answer)

        is_correct = model_answer_bool == answer_mapping[question_type]

        bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()

        if qtype == 'positive' or qtype == 'negative_false':
          mentioned_colour = col1
        else:
          mentioned_colour = col2
        if qtype == 'positive' or qtype == 'positive_false':
          mentioned_attn_pos.append(bbox_attentions[mentioned_colour])
          unmentioned_attn_pos.append(bbox_attentions[unmentioned_colour])
        else:
          mentioned_attn_neg.append(bbox_attentions[mentioned_colour])
          unmentioned_attn_neg.append(bbox_attentions[unmentioned_colour])


np.save(save_dir + f"/{model_name}_{image_dir}_{save_neg}mentioned_attn.npy", np.array(mentioned_attn))
np.save(save_dir + f"/{model_name}_{image_dir}_{save_neg}non_mentioned_attn.npy", np.array(non_mentioned_attn))
