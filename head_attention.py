from plotter import Plotter
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import InternVLForConditionalGeneration
from transformers import LlavaForConditionalGeneration


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
    

visualise_attention("paligemma", "2d_dataset_fixed_positions_1000/images/image_0000.png")
