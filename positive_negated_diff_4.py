from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import multiprocessing as mp
from transformers import AutoModelForImageTextToText, InternVLForConditionalGeneration
import os

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
                model_id, torch_dtype=torch.float16).to(device)
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
                model_id, torch_dtype=torch.float16).to(device)
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
                model_id, torch_dtype=torch.float16).to(device)
            _loaded_models[model_id] = (processor, model)
        else:
            print("Reusing cached model and processor.")
            processor, model = _loaded_models[model_id]


    return processor, model

def plot_diff(model_name):

    processor, model = load_model(model_name)

    mentioned_diff_attention = []
    not_mentioned_diff_attention = []
    baseline_attentions = []
    for i in range(100):
        image_path = f"4_shapes_same_dataset_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
        top_left_colour = plotter.get_shape_by_position('top_left').colour
        top_right_colour = plotter.get_shape_by_position('top_right').colour
        bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
        bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour

        for target_colour in [top_left_colour, top_right_colour, bottom_left_colour, bottom_right_colour]:


            plotter.set_model(model, processor)
            plotter.get_outputs(f"The figure is {target_colour}")
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            # item1_attn_pos = bbox_attentions[top_left_colour]
            # item2_attn_pos = bbox_attentions[top_right_colour]
            # item3_attn_pos = bbox_attentions[bottom_left_colour]
            # item4_attn_pos = bbox_attentions[bottom_right_colour]

            mentioned_attention_pos = bbox_attentions[target_colour]
            not_mentioned_attention_pos = [attn for colour, attn in bbox_attentions.items() if colour != target_colour]
            # baseline_attentions.append(baseline_attn)
            plotter.get_outputs(f"The figure is not {target_colour}")
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            # item1_attn_neg = bbox_attentions[top_left_colour]
            # item2_attn_neg = bbox_attentions[top_right_colour]
            # item3_attn_neg = bbox_attentions[bottom_left_colour]
            # item4_attn_neg = bbox_attentions[bottom_right_colour]

            mentioned_attention_neg = bbox_attentions[target_colour]
            not_mentioned_attention_neg = [attn for colour, attn in bbox_attentions.items() if colour != target_colour]

            # baseline_attentions.append(baseline_attn)
            # calculate differnece betwen item1_attn_pos and item1_attn_neg
            mentioned_diff_attention.append(np.array(mentioned_attention_pos) - np.array(mentioned_attention_neg))
            not_mentioned_diff_attention.append(np.mean(np.array(not_mentioned_attention_pos) - np.array(not_mentioned_attention_neg), axis=0))

    # plot difference between positive and negated attention 
    avg_item1_attn = np.mean(np.array(mentioned_diff_attention), axis=0)
    avg_item2_attn = np.mean(np.array(not_mentioned_diff_attention), axis=0)
    plt.figure(figsize=(10, 6))
    # plt.plot(layers, avg_item1_attn, label='Left Shape Pos-Neg Diff')
    # plt.plot(layers, avg_item2_attn, label='Right Shape Pos-Neg Diff')
    # #plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    # plt.xlabel('Layer')
    # plt.ylabel('Attention')
    # plt.title('Average Attention Scores Through Layers')
    # plt.legend()
    # plt.savefig(f'positive_negated_diff_test.png')
    # plot bar graph of difference
    x = np.array(list(range(len(avg_item1_attn))))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, avg_item1_attn, width, label=f"Mentioned")
    ax.bar(x - 0.5*width, avg_item2_attn, width, label=f"Not Mentioned")
    for i in range(len(avg_item1_attn)):
        ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Layer')
    # set ylim based on max value of both item1 and item2
    max_attn = max(max(avg_item1_attn), max(avg_item2_attn))
    ax.set_ylim(-max_attn * 1.1, max_attn * 1.1)
    ax.set_ylabel('Attention Difference')
    ax.set_title('Average Attention Difference (positive - Negated) Through Layers')
    ax.legend()
    print(f"saving to plots/positive_negative_diff_graphs/4_shape_{model_name}_positive_negated_diff.png")
    plt.savefig(f'plots/positive_negative_diff_graphs/4_shape_{model_name}_positive_negated_diff.png')
if __name__ == "__main__":
    img_folder = "4_shapes_same_dataset_1000"
    models = ['llava', 'internvl', 'paligemma']
    # mp.set_start_method('spawn', force=True)
    for model in models:
        print(f"Processing model: {model}")
        p = mp.Process(target=plot_diff, args=(model,))
        p.start()
        p.join()




