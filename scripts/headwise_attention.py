import matplotlib.pyplot as plt
import numpy as np
from functions import load_model, mean_and_ci
import torch
import gc
import multiprocessing as mp
from plotter import Plotter
import argparse
import seaborn as sns
from pathlib import Path


def plot_headwise(model_name, image_path):
    print("loading model ", model_name)
    processor, model = load_model(model_name)
    if "whatsup" in image_path:
      noun = "object"
    else:
      noun = "shape"
    plotter = Plotter(image_path)
    plotter.set_model(model, processor)
    left_colour = plotter.get_left_shapes()[0].colour
    right_colour = plotter.get_right_shapes()[0].colour
    plotter.get_outputs(f"The {noun} is {left_colour}")
    
    pos_attentions = plotter.plot_headwise_attention()
    #print(bbox_attentions[left_colour])
    pos_mentioned = np.array(pos_attentions[left_colour])

    plotter = Plotter(image_path)
    plotter.set_model(model, processor)
    left_colour = plotter.get_left_shapes()[0].colour
    right_colour = plotter.get_right_shapes()[0].colour
    plotter.get_outputs(f"The {noun} is not {left_colour}")
    
    neg_attentions = plotter.plot_headwise_attention()
    neg_mentioned = np.array(neg_attentions[left_colour])

    print(f"Data type: {type(pos_mentioned)}")
    print(f"Data shape: {pos_mentioned.shape}")
    print(f"Argmax result: {np.argmax(pos_mentioned)}")


    max_layer_pos, max_head_pos = np.unravel_index(np.argmax(pos_mentioned), pos_mentioned.shape)
    max_layer_neg, max_head_neg = np.unravel_index(np.argmax(neg_mentioned), neg_mentioned.shape)
    print(f"For image {image_path}:\n max pos attention at layer {max_layer_pos}, head {max_head_pos}\n max neg attention at layer {max_layer_neg}, head {max_head_neg}")
    data = neg_mentioned - pos_mentioned

    sns.heatmap(data, annot=False, xticklabels=range(len(data[0])), yticklabels=range(len(data)))
    plt.xlabel("Heads")
    plt.ylabel("Layers")
    plt.title(f"Attention Difference for Mentioned")

    path_str = "binary/images/image_0000.png"
    filename = Path(path_str).stem
    plt.savefig(f"attention_diffs/{filename}.png")
    plt.close()



if __name__ == "__main__":

    plot_headwise("llava", "binary/images/image_0000.png")
    plot_headwise("llava", "binary/images/image_0001.png")






