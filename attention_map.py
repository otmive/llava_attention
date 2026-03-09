from plotter import Plotter
import torch 
import os 
from transformers import AutoProcessor, AutoModelForImageTextToText, InternVLForConditionalGeneration, LlavaForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import gc
from matplotlib import rcParams
import math

def load_model(model_name):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global _loaded_models
    if '_loaded_models' not in globals():
        _loaded_models = {}

    if model_name == "internvl":
        model_id = "OpenGVLab/InternVL3_5-4B-HF"
        if model_id not in _loaded_models:
            print("Loading model and processor...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = InternVLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, attn_implementation="eager").to(device)
            _loaded_models[model_id] = (processor, model)
        else:
            print("Reusing cached model and processor.")
            processor, model = _loaded_models[model_id]
    
    elif model_name == "llava":
        model_id = "llava-hf/llava-1.5-7b-hf"
        if model_id not in _loaded_models:
            print("Loading model and processor...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, attn_implementation="eager").to(device)
            _loaded_models[model_id] = (processor, model)
        else:
            print("Reusing cached model and processor.")
            processor, model = _loaded_models[model_id]

    elif model_name == "paligemma":

        model_id = "google/paligemma2-3b-mix-224"
        if model_id not in _loaded_models:
            print("Loading model and processor...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.float16, attn_implementation="eager").to(device)
            _loaded_models[model_id] = (processor, model)
        else:
            print("Reusing cached model and processor.")
            processor, model = _loaded_models[model_id]


    return processor, model


def plot_attention_map(model_name, image_path, prompt):

    print("loading model ", model_name)
    processor, model = load_model(model_name)
    

    plotter = Plotter(image_path)
    left_colour = plotter.get_left_shapes()[0].colour
    right_colour = plotter.get_right_shapes()[0].colour
    plotter.set_model(model, processor)
    plotter.get_outputs(prompt)

    cols = 5

    # plot a figure for each layer with heads arrayed on a plot (sub plots) and save
    for layer in range(len(plotter.outputs["attentions"][0])):
        heads = len(plotter.outputs["attentions"][0][0][0])
        print(f"Layer {layer} has {heads} heads")
        rows = math.ceil(heads / cols)
        # create a figure with subplots for each head in the layer
        fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        fig.suptitle(f"Layer {layer} Attention Maps for prompt: {prompt}", fontsize=16)
        # print(plotter.outputs['attentions'][0][layer].shape)




        for head in range(heads):
            attn = plotter.get_image_attention_matrix(layer, head)
            ax = axs.flatten()[head]
            # add numerical values to the squares in the attention map
            for i in range(attn.shape[0]):
                for j in range(attn.shape[1]):
                    ax.text(j, i, f"{attn[i, j]:.4f}", ha='center', va='center', fontsize=4)
            im = ax.imshow(attn, cmap='viridis')
            ax.set_title(f"Head {head}")
            ax.axis('off')
        plt.tight_layout()
        save_path = f"{model_name}_attention_maps/layer_{layer}_{image_path.split('/')[-1].split('.')[0]}_{prompt.replace(' ', '_')}.png"
        plt.savefig(save_path)
        plt.close()


def plot_attention_map_for_layers(model_name, image_path, prompt):
    
    print("loading model ", model_name)
    processor, model = load_model(model_name)
    

    plotter = Plotter(image_path)
    left_colour = plotter.get_left_shapes()[0].colour
    right_colour = plotter.get_right_shapes()[0].colour
    plotter.set_model(model, processor)
    plotter.get_outputs(prompt)

    cols = 5

    num_layers = len(plotter.outputs["attentions"][0])
    rows = math.ceil(num_layers / cols)

    # create a figure with subplots for each layer
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    fig.suptitle(f"Attention Maps for prompt: {prompt}", fontsize=16)
    
    for layer in range(num_layers):
        attn = plotter.get_image_attention_matrix(layer)  # Get attention for the first head
        ax = axs.flatten()[layer]
        # add numerical values to the squares in the attention map
        for i in range(attn.shape[0]):
            for j in range(attn.shape[1]):
                ax.text(j, i, f"{attn[i, j]:.4f}", ha='center', va='center', fontsize=4)
        im = ax.imshow(attn, cmap='viridis')
        ax.set_title(f"Layer {layer}")
        ax.axis('off')

    # Hide any unused subplots if num_layers < rows * cols
    for i in range(num_layers, len(axs.flatten())):
        axs.flatten()[i].axis('off')

    plt.tight_layout()
    save_path = f"{model_name}_attention_maps/layers_{image_path.split('/')[-1].split('.')[0]}_{prompt.replace(' ', '_')}.png"
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    image_path = "2d_dataset_fixed_positions/images/image_0000.png"
    model_name = "paligemma"
    prompt = "The figure is yellow"
    plot_attention_map(model_name, image_path, prompt)