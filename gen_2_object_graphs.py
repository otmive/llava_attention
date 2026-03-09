from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration
from transformers import AutoModelForImageTextToText
import os 
import argparse

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

def four_position(model_name, image_folder, pos, colour=None):

    processor, model = load_model(model_name)
    count = 0
    item1_attn = []
    item2_attn = []
    baseline_attentions = []
    all_files = os.listdir(f"{image_folder}/images")
    files_to_process = all_files[:100] if len(all_files) >= 1000 else all_files
    for filename in files_to_process:
        image_path = f"{image_folder}/images/{filename}"
        print(f"Processing image: {image_path}")
        plotter = Plotter(image_path)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour
        plotter.set_model(model, processor)

        if pos == 'left' or pos == 'neg_left':
            target_colour = left_colour
        elif pos == 'right' or pos == 'neg_right':
            target_colour = right_colour

        if target_colour == colour or colour is None:

            if pos == 'left':
                plotter.get_outputs(f"The object is a {left_colour}")
            elif pos == 'right':
                plotter.get_outputs(f"The object is a {right_colour}")
            elif pos == 'neg_left':
                plotter.get_outputs(f"The object is not a {left_colour}")
            elif pos == 'neg_right':
                plotter.get_outputs(f"The object is not a {right_colour}")

            print("outputs:", plotter.print_output())
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            item1_attn.append(bbox_attentions[left_colour])
            item2_attn.append(bbox_attentions[right_colour])
            baseline_attentions.append(baseline_attn)
            count += 1

    # plot average attention across items
    avg_item1_attn = np.mean(np.array(item1_attn), axis=0)
    avg_item2_attn = np.mean(np.array(item2_attn), axis=0)
    avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)
    std_item1_attn = np.std(np.array(item1_attn), axis=0)
    std_item2_attn = np.std(np.array(item2_attn), axis=0)

    ci1 = 1.96*(std_item1_attn/np.sqrt(len(avg_item1_attn)))
    ci2 = 1.96*(std_item2_attn/np.sqrt(len(avg_item2_attn)))

    if model_name == 'llava':
        num_layers = 32
    elif model_name == 'internvl':
        num_layers = 36
    elif model_name == 'paligemma':
        num_layers = 26
    layers = list(range(num_layers))

    # plt.figure(figsize=(10, 6))
    # plt.plot(layers, avg_item1_attn, label='Left Shape Attention')
    # plt.plot(layers, avg_item2_attn, label='Right Shape Attention')
    # #plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    # # plot confidence intervals
    # plt.fill_between(layers, avg_item1_attn - std_item1_attn, avg_item1_attn + std_item1_attn, color='blue', alpha=0.2)
    # plt.fill_between(layers, avg_item2_attn - std_item2_attn, avg_item2_attn + std_item2_attn, color='red', alpha=0.2)    
    # plt.xlabel('Layer')
    # plt.ylabel('Attention')
    # plt.title('Average Attention Scores Through Layers')
    # plt.legend()
    # # fix y axis limit
    # #plt.ylim(0, 0.018)
    # #plt.ylim(0, 0.005)
    # plt.ylim(0, ylim)
    # plt.savefig(f'{save_folder}/{pos}_{colour+"_" if colour else ""}{model_name}.png')
    # print(f"Processed {count} images matching criteria.")
    return avg_item1_attn, avg_item2_attn, ci1, ci2


import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

def run_model_analysis(model_name, img_folder, positions):
    """
    Runs in a dedicated process to ensure VRAM is cleared upon completion.
    """
    print(f"--- Starting Process: {model_name} ---")
    
    # 1. Collect data for all positions first
    results = []
    for pos in positions:
        print(f"[{model_name}] Processing position: {pos}")
        # Assuming four_position returns two lists/arrays of attention scores
        item1_attn, item2_attn, ci1, ci2 = four_position(model_name, img_folder, pos)
        results.append((pos, item1_attn, item2_attn, ci1, ci2))
    
    # 2. Determine the global maximum for this model to set the Y-limit
    # We flatten the values to find the absolute max across all positions and items
    all_values = []
    for _, i1, i2, ci1, ci2 in results:
        all_values.extend(i1 + ci1)
        all_values.extend(i2 + ci2)
    
    global_max = max(all_values) if all_values else 1.0
    y_limit = global_max * 1.1  # Adding 10% padding for visual clarity

    title_map = {
        "left": "Positive Left",
        "right": "Positive Right",
        "neg_left": "Negated Left",
        "neg_right": "Negated Right"
    }
    # 3. Create the figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    fig.suptitle(f"{model_name.upper()}", fontsize=16, fontweight='bold')
    
    for i, (pos, item1_attn, item2_attn, ci1, ci2) in enumerate(results):
        x = list(range(len(item1_attn)))
        axes[i].plot(item1_attn, label='Left Shape', linewidth=2, marker='o', markersize=3)
        axes[i].plot(item2_attn, label='Right Shape', linewidth=2, marker='x', markersize=3)
        
        axes[i].fill_between(x, item1_attn - ci1, item1_attn + ci1, alpha=0.25)
        axes[i].fill_between(x, item2_attn - ci2, item2_attn + ci2, alpha=0.25)

        axes[i].set_title(f"{title_map.get(pos, pos)}")
        axes[i].set_xlabel("Layer")
        axes[i].set_ylim(0, y_limit)  # Apply the calculated max limit
        
        if i == 0:
            axes[i].set_ylabel("Attention")
        
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save results
    output_filename = f"plots/2d_datasets_2/{model_name}_position_graphs.png"
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"--- Finished {model_name}. Plot saved as {output_filename} ---")

if __name__ == "__main__":
    img_folder = "2d_dataset_fixed_positions_1000"
    #img_folder = "whatsup"
    models = ["llava", "internvl", "paligemma"]
    positions = ["left", "neg_left", "right", "neg_right"]

    # Use 'spawn' for cleaner GPU memory management if on Linux/Mac
    # mp.set_start_method('spawn', force=True)

    for model in models:
        # We start a fresh process for each model
        p = mp.Process(target=run_model_analysis, args=(model, img_folder, positions))
        p.start()
        
        # p.join() is critical: it makes the script wait for the process to die 
        # (and free VRAM) before starting the next model.
        p.join()