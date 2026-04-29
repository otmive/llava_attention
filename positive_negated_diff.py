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


def plot_diff(model_name, image_folder):

    processor, model = load_model(model_name)

    item1_diff_attn = []
    item2_diff_attn = []
    baseline_attentions = []
    with torch.inference_mode():
        for image_path in os.listdir(f"{image_folder}/images"):
            plotter = Plotter(f"{image_folder}/images/{image_path}")
            left_colour = plotter.get_left_shapes()[0].colour
            right_colour = plotter.get_right_shapes()[0].colour
            plotter.set_model(model, processor)
            print(f"The figure is {left_colour}")
            ## LEFT
            plotter.get_outputs(f"The object is a {left_colour}")
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            # item 1 is mentioned colour
            item1_attn_pos = bbox_attentions[left_colour]
            item2_attn_pos = bbox_attentions[right_colour]
            # baseline_attentions.append(baseline_attn)
            #make prmopt in all caps
            plotter.get_outputs(f"The object is not a {left_colour}")
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            item1_attn_neg = bbox_attentions[left_colour]
            item2_attn_neg = bbox_attentions[right_colour]
            # baseline_attentions.append(baseline_attn)
            # calculate differnece betwen item1_attn_pos and item1_attn_neg
            item1_diff_attn.append(np.array(item1_attn_pos) - np.array(item1_attn_neg))
            item2_diff_attn.append(np.array(item2_attn_pos) - np.array(item2_attn_neg))

            ## RIGHT
            plotter.get_outputs(f"The object is a {right_colour}")
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            # item 1 is mentinoed colour
            item1_attn_pos = bbox_attentions[right_colour]
            item2_attn_pos = bbox_attentions[left_colour]
            # baseline_attentions.append(baseline_attn)
            #make prmopt in all caps
            plotter.get_outputs(f"The object is not a {right_colour}")
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            item1_attn_neg = bbox_attentions[right_colour]
            item2_attn_neg = bbox_attentions[left_colour]
            # baseline_attentions.append(baseline_attn)
            # calculate differnece betwen item1_attn_pos and item1_attn_neg
            item1_diff_attn.append(np.array(item1_attn_pos) - np.array(item1_attn_neg))
            item2_diff_attn.append(np.array(item2_attn_pos) - np.array(item2_attn_neg))


    # plot difference between positive and negated attention 
    avg_item1_attn = np.mean(np.array(item1_diff_attn), axis=0)
    avg_item2_attn = np.mean(np.array(item2_diff_attn), axis=0)
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
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, avg_item1_attn, width, label='Mentioned')
    ax.bar(x + width/2, avg_item2_attn, width, label='Not Mentioned')
    # set x axis
    # set ylim based on max value of both item1 and item2
    max_attn = max(max(avg_item1_attn), max(avg_item2_attn), max(-min(avg_item1_attn), -min(avg_item2_attn)))
    ax.set_ylim(-max_attn * 1.1, max_attn * 1.1)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Attention Difference')
    ax.set_title('Average Attention Difference (Positive - Negative) Through Layers')
    ax.legend()
    plt.savefig(f'plots/positive_negative_diff_graphs/whatsup_{model_name}_all.png')
    print(f"Saved plot at file: plots/positive_negative_diff_graphs/whatsup_{model_name}_all.png")
if __name__ == "__main__":
    img_folder = "whatsup"
    # #img_folder = "whatsup"
    models = ["llava", "internvl", "paligemma"]
    positions = ["left", "right"]

    # Use 'spawn' for cleaner GPU memory management if on Linux/Mac
    # mp.set_start_method('spawn', force=True)

    for model in models:
        # We start a fresh process for each model
        p = mp.Process(target=plot_diff, args=(model, img_folder))
        p.start()
        
        # p.join() is critical: it makes the script wait for the process to die 
        # (and free VRAM) before starting the next model.
        p.join()

 



