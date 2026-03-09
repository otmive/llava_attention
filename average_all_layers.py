from plotter import Plotter
import torch 
import os 
from transformers import AutoProcessor, AutoModelForImageTextToText, InternVLForConditionalGeneration, LlavaForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import gc

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

def plot_all_layers_2(model_name, data_dir):
    item1_attn = []
    item2_attn = []
    processor, model = load_model(model_name)

    for img_path in os.listdir(data_dir)[0:2]:
        print("image path: ", img_path)
        plotter = Plotter(data_dir + "/" + img_path)
        plotter.set_model(model, processor)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour

        plotter.get_outputs(f"The figure is {left_colour}")
        bbox_attentions, _ = plotter.plot_attention_through_layers(save_folder=f"bar_plots/test_{model_name}.png")
        item1_attns = bbox_attentions[left_colour]
        item2_attns = bbox_attentions[right_colour]
        print("item1_attns: ", item1_attns)
        print("item2_attns: ", item2_attns)
        item1_attn.append(np.mean(item1_attns))
        item2_attn.append(np.mean(item2_attns))

    # plot average attention for left and right shapes
    labels = ['Left Shape', 'Right Shape']
    means = [np.mean(item1_attn), np.mean(item2_attn)]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, means, width, label='Attention')
    ax.set_ylabel('Average Attention')
    ax.set_title('Average Attention by Shape Position')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    fig.tight_layout()
    plt.savefig(f"bar_plots/average_attention_{model_name}.png")
    plt.close()

     # free up memory
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Finished plotting for ", model_name)


def plot_all_layers_4(model_name, data_dir, save_path):
    item1_attn_left = []
    item2_attn_left = []
    item1_attn_neg_left = []
    item2_attn_neg_left = []
    item1_attn_right = []
    item2_attn_right = []
    item1_attn_neg_right = []
    item2_attn_neg_right = []
    processor, model = load_model(model_name)

    for pos in ['left', 'right', 'neg_left', 'neg_right']:
        item1_attn = []
        item2_attn = []
        for img_path in os.listdir(data_dir)[0:5]:
            print("image path: ", img_path)
            plotter = Plotter(data_dir + "/" + img_path)
            plotter.set_model(model, processor)
            left_colour = plotter.get_left_shapes()[0].colour
            right_colour = plotter.get_right_shapes()[0].colour

            if pos == 'left':
                plotter.get_outputs(f"The figure is {left_colour}")
            elif pos == 'right':
                plotter.get_outputs(f"The figure {right_colour}")
            elif pos == 'neg_left':
                plotter.get_outputs(f"The figure is not {left_colour}")
            elif pos == 'neg_right':
                plotter.get_outputs(f"The figure is not {right_colour}")

            bbox_attentions, _ = plotter.plot_attention_through_layers(save_path=f"bar_plots/test_{pos}.png")
            item1_attns = bbox_attentions[left_colour]
            item2_attns = bbox_attentions[right_colour]
            print("item1_attns: ", item1_attns)
            print("item2_attns: ", item2_attns)
        if pos == 'left':
            item1_attn_left.append(np.mean(item1_attns))
            item2_attn_left.append(np.mean(item2_attns))
        elif pos == 'right':
            item1_attn_right.append(np.mean(item1_attns))
            item2_attn_right.append(np.mean(item2_attns))
        elif pos == 'neg_left':
            item1_attn_neg_left.append(np.mean(item1_attns))
            item2_attn_neg_left.append(np.mean(item2_attns))
        elif pos == 'neg_right':
            item1_attn_neg_right.append(np.mean(item1_attns))
            item2_attn_neg_right.append(np.mean(item2_attns))


    # plot average attention for each of the four types of prompt for lef tnad right shapes
    labels = ['left', 'neg_left', 'right', 'neg_right']
    item1_means = [np.mean(item1_attn_left), np.mean(item1_attn_neg_left), np.mean(item1_attn_right), np.mean(item1_attn_neg_right)]
    item2_means = [np.mean(item2_attn_left), np.mean(item2_attn_neg_left), np.mean(item2_attn_right), np.mean(item2_attn_neg_right)]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, item1_means, width, label='Left Shape Attention')
    rects2 = ax.bar(x + width/2, item2_means, width, label='Right Shape Attention')
    ax.set_ylabel('Average Attention')
    ax.set_title('Average Attention by Prompt Type and Shape Position')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

     # free up memory
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Finished plotting for ", model_name)

import numpy as np
import matplotlib.pyplot as plt
import gc
import torch
import os

def plot_all_layers_4(model_name, data_dir, save_path):
    positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 
                 'neg_top_left', 'neg_top_right', 'neg_bottom_left', 'neg_bottom_right']
    
    data_store = {pos: [[] for _ in range(4)] for pos in positions}
    processor, model_obj = load_model(model_name)

    # 1. Process data
    for pos in positions:
        # Get only image files to avoid directory errors
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_path in image_files[0:5]:
            try:
                print(f"Processing: Position={pos}, Image={img_path}")
                plotter = Plotter(os.path.join(data_dir, img_path))
                plotter.set_model(model_obj, processor)
                
                shapes = {p: plotter.get_shape_by_position(p).colour for p in ['top_left', 'top_right', 'bottom_left', 'bottom_right']}
                
                base_pos = pos.replace('neg_', '')
                is_negative = pos.startswith('neg_')
                prompt = f"The figure is {'not ' if is_negative else ''}{shapes[base_pos]}"
                
                plotter.get_outputs(prompt)
                print("OU?TPUTS: ", plotter.print_output())
                bbox_attentions, baseline_attentions = plotter.plot_attention_through_layers(f"plots/test_{model_name}_thru_layers")

                # Set layers
                if 'llava' in model_name.lower():
                    key_layers = list(range(15, 27))
                elif 'internvl' in model_name.lower():
                    key_layers = list(range(20, 35))
                elif 'paligemma' in model_name.lower():
                    key_layers = list(range(11, 24))
                else:
                    key_layers = [0]

                # 2. THE FIX: Cast to np.array before indexing with a list
                for i, p_key in enumerate(['top_left', 'top_right', 'bottom_left', 'bottom_right']):
                    color = shapes[p_key]
                    # Ensure we have the color in the results
                    if color in bbox_attentions:
                        attn_list = np.array(bbox_attentions[color])
                        attn_val = np.mean(attn_list[key_layers])
                        data_store[pos][i].append(attn_val)
                    
            except Exception as e:
                print(f"Error on {img_path} at {pos}: {e}")
                continue

    # 3. Calculate Final Means (using nanmean to handle empty/failed runs)
    final_means = {}
    for pos in positions:
        pos_results = []
        for item_list in data_store[pos]:
            if len(item_list) > 0:
                pos_results.append(np.mean(item_list))
            else:
                pos_results.append(0.0) # No data collected
        final_means[pos] = pos_results

    print("Final aggregated means:", final_means)

    # 4. Plotting
    if not any(any(v) for v in final_means.values()):
        print("No data collected to plot!")
        return

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    item_labels = ['Top Left Shape', 'Top Right Shape', 'Bottom Left Shape', 'Bottom Right Shape']
    display_positions = ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right']
    x = np.arange(2) 
    width = 0.15

    for i in range(4):
        pos_key = positions[i]       
        neg_key = positions[i+4]     
        
        for item_idx in range(4):
            offset = (item_idx - 1.5) * width
            vals = [final_means[pos_key][item_idx], final_means[neg_key][item_idx]]
            axs[i].bar(x + offset, vals, width, label=item_labels[item_idx])

        axs[i].set_title(f'Attention at {display_positions[i]} Position')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(['Positive', 'Negative'])
        axs[i].legend()

    plt.tight_layout()
    
    # Ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    # 5. Cleanup
    del model_obj
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Finished plotting for {model_name}")


if __name__ == "__main__":
    model_names = ['llava', 'internvl', 'paligemma']
    
    for model_name in model_names:
        print(f"--- Starting Process for {model_name} ---")
        
        # Define the process
        p = mp.Process(
            target=plot_all_layers_4, 
            args=(model_name, "4_shapes_same_dataset_1000/images", f"bar_plots/average_attention_4_shapes_{model_name}_test.png")
        )
        
        p.start()
        p.join()  # Wait for this model to finish entirely
        
        print(f"--- Process for {model_name} exited with code: {p.exitcode} ---")
        
        # If exitcode is not 0, it crashed (likely OOM)
        if p.exitcode != 0:
            print(f"Warning: {model_name} failed. Moving to next model.")

    # plot_all_layers("llava", "2d_dataset_fixed_positions/images", f"bar_plots/test_llava.png")