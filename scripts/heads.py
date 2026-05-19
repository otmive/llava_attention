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


processor, model = load_model("paligemma")
image_path = "binary/images/image_0000.png"
print(f"Processing image: {image_path}")
plotter = Plotter(image_path)
left_colour = plotter.get_left_shapes()[0].colour
right_colour = plotter.get_right_shapes()[0].colour
plotter.set_model(model, processor)

prob = plotter.get_outputs_prob(f"Is this statement correct? The object on the left is {left_colour}. Answer yes or no.", "yes")
print("Orig Prob: ", prob)

# for l in range()
prob = plotter.get_outputs_prob(f"Is this statement correct? The object on the left is {left_colour}. Answer yes or no.", "yes", 0,0)
print("Prob: ", prob)