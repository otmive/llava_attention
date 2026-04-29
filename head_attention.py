from plotter import Plotter
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import InternVLForConditionalGeneration
from transformers import LlavaForConditionalGeneration
import os 
import numpy as np 
import matplotlib.pyplot as plt

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

def visualise_attention(model_name, image_path):

    processor, model = load_model(model_name)

    plotter = Plotter(image_path)
    plotter.set_model(model, processor)
    plotter.get_outputs("The figure is green")


    # make a dictionary to store each layer and head value
    attention_dict_pos = {}

    # print out total attention for each head at each layer
    for layer in range(len(plotter.outputs["attentions"][0])):
        print(plotter.outputs['attentions'][0][layer].shape)
        for head in range(len(plotter.outputs["attentions"][0][0][0])):
            total_attention = plotter.get_image_attention_matrix(layer, head).sum().item()
            print(f"Layer {layer}, Head {head}, Total Attention: {total_attention}")
            attention_dict_pos[(layer, head)] = total_attention

    plotter = Plotter(image_path)
    plotter.set_model(model, processor)
    plotter.get_outputs("The figure is not green")


    # make a dictionary to store each layer and head value
    attention_dict_neg = {}

    # print out total attention for each head at each layer
    for layer in range(len(plotter.outputs["attentions"][0])):
        print(plotter.outputs['attentions'][0][layer].shape)
        for head in range(len(plotter.outputs["attentions"][0][0][0])):
            total_attention = plotter.get_image_attention_matrix(layer, head).sum().item()
            print(f"Layer {layer}, Head {head}, Total Attention: {total_attention}")
            attention_dict_neg[(layer, head)] = total_attention
    
    # plot with layer on x axis and head on y axis heatmap 
    import matplotlib.pyplot as plt
    import numpy as np

    # plot attentiondict_pos and save heatmap to file
    layers = sorted(set([key[0] for key in attention_dict_pos.keys()]))
    heads = sorted(set([key[1] for key in attention_dict_pos.keys()]))
    attention_values_pos = np.zeros((len(layers), len(heads)))
    for layer in layers:
        for head in heads:
            attention_values_pos[layer][head] = attention_dict_pos[(layer, head)]
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_values_pos, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Total Attention')
    plt.xticks(ticks=range(len(heads)), labels=[f'Head {head}' for head in heads], rotation=45)
    plt.yticks(ticks=range(len(layers)), labels=[f'Layer {layer}' for layer in layers])
    plt.title(f'Total Attention from Image Tokens to Text Tokens for {model_name} (Positive)')
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    plt.tight_layout()
    plt.savefig(f'{model_name}_attention_heatmap_positive.png')

    # plot attentiondict_neg and save heatmap to file
    attention_values_neg = np.zeros((len(layers), len(heads)))
    for layer in layers:
        for head in heads:
            attention_values_neg[layer][head] = attention_dict_neg[(layer, head)]
    plt.figure(figsize=(10, 8))
    # set heatmap scale to be between 0 and 1
    
    plt.imshow(attention_values_neg, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Total Attention')
    plt.xticks(ticks=range(len(heads)), labels=[f'Head {head}' for head in heads], rotation=45)
    plt.yticks(ticks=range(len(layers)), labels=[f'Layer {layer}' for layer in layers])
    plt.title(f'Total Attention from Image Tokens to Text Tokens for {model_name} (Negative)')
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    plt.tight_layout()
    plt.savefig(f'{model_name}_attention_heatmap_negative.png')




    layers = sorted(set([key[0] for key in attention_dict_pos.keys()]))
    heads = sorted(set([key[1] for key in attention_dict_pos.keys()]))
    # calculate different between positive and negative attention
    attention_values = np.zeros((len(layers), len(heads)))
    for layer in layers:
        for head in heads:
            attention_values[layer][head] = np.abs(attention_dict_pos[(layer, head)] - attention_dict_neg[(layer, head)])

    plt.figure(figsize=(10, 8))
    plt.imshow(attention_values, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Total Attention')
    plt.xticks(ticks=range(len(heads)), labels=[f'Head {head}' for head in heads], rotation=45)
    plt.yticks(ticks=range(len(layers)), labels=[f'Layer {layer}' for layer in layers])
    plt.title(f'Total Attention from Image Tokens to Text Tokens for {model_name}')
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    plt.tight_layout()
    plt.savefig(f'{model_name}_attention_heatmap_diff.png')
    

def visualise_heads_all_images(model_name, image_dir, save_dir):
    
    processor, model = load_model(model_name)

    attention_sums = None
    count = 0
    neg_attention_sums = None
    neg_count = 0
    for image_path in os.listdir(image_dir)[0:10]:
        plotter = Plotter(os.path.join(image_dir, image_path))
        plotter.set_model(model, processor)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour
        for pos in ['left']:
            if pos == "left":
                target_colour = left_colour
            else:
                target_colour = right_colour
            plotter.get_outputs(f"The figure is {target_colour}")
            num_layers = len(plotter.outputs["attentions"][0])
            num_heads = len(plotter.outputs["attentions"][0][0][0])
            for layer in range(num_layers):
                for head in range(num_heads):
                    if attention_sums is None:
                        sample_matrix = plotter.get_image_attention_matrix(0, 0)
                        H, W = sample_matrix.shape
                        attention_sums = np.zeros((num_layers, num_heads, H, W))
                    image_attention = plotter.get_image_attention_matrix(layer, head)
                    attention_sums[layer, head] += image_attention
            count+=1

        plotter = Plotter(os.path.join(image_dir, image_path))
        plotter.set_model(model, processor)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour
        for pos in ['left']:
            if pos == "left":
                target_colour = left_colour
            else:
                target_colour = right_colour
            plotter.get_outputs(f"The figure is not {target_colour}")
            num_layers = len(plotter.outputs["attentions"][0])
            num_heads = len(plotter.outputs["attentions"][0][0][0])
            for layer in range(num_layers):
                for head in range(num_heads):
                    if neg_attention_sums is None:
                        sample_matrix = plotter.get_image_attention_matrix(0, 0)
                        H, W = sample_matrix.shape
                        neg_attention_sums = np.zeros((num_layers, num_heads, H, W))
                    image_attention = plotter.get_image_attention_matrix(layer, head)
                    neg_attention_sums[layer, head] += image_attention
            neg_count+=1

        

    print("total shape")
    print(attention_sums.shape)
    #get average W,H array for each layer and head (avged over images)
    attention_averages = attention_sums / count 
    print("average shape")
    print(attention_averages.shape)
    # plot attention_averages and save heatmap to file
    neg_attention_averages = neg_attention_sums / neg_count
    vmax = 0
    key_layers = [11,12,13,14,15,16,17,18,19,20]
    for layer in key_layers:
        num_heads = attention_averages.shape[1]
        
        # We want Pairs: [Pos, Neg] [Pos, Neg]
        # Let's keep the grid structure but double the columns for the side-by-side view
        num_cols_pairs = 2 
        num_rows = num_heads # One row per head for clarity, or adjust if you have many heads
        
        fig, axes = plt.subplots(num_rows, num_cols_pairs, figsize=(10, num_rows * 4))
        
        # Ensure axes is 2D even if there's only one head
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for head in range(num_heads):
            # Determine global max for consistent scaling across both plots
            vmax = max(attention_averages[layer, head].max(), neg_attention_averages[layer, head].max())
            
            # Left: Attention Averages
            im_pos = axes[head, 0].imshow(attention_averages[layer, head], cmap='viridis', vmin=0, vmax=vmax)
            axes[head, 0].set_title(f'Layer {layer}, Head {head} (Pos)')
            axes[head, 0].axis('off')
            
            # Right: Neg Attention Averages
            im_neg = axes[head, 1].imshow(neg_attention_averages[layer, head], cmap='viridis', vmin=0, vmax=vmax)
            axes[head, 1].set_title(f'Layer {layer}, Head {head} (Neg)')
            axes[head, 1].axis('off')

        plt.tight_layout()
        
        # Add a colorbar that references the last plotted image
        cbar = fig.colorbar(im_neg, ax=axes.ravel().tolist(), shrink=0.5)
        cbar.set_label('Attention')

        
        save_path = os.path.join(save_dir, f'{model_name}_comparison_layer_{layer}.png')
        plt.savefig(save_path)
        plt.close() # Close to free up memory
visualise_heads_all_images("paligemma", "2d_dataset_fixed_positions_1000/images", "plots/head_heatmaps")
