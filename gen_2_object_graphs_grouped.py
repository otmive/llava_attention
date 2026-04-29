from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration
from transformers import AutoModelForImageTextToText
import os 
import argparse
import multiprocessing as mp

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

def get_pos_results(model_name, image_folder, ylim, test=False, save_folder=None, colour=None):

    processor, model = load_model(model_name)
    count = 0
    mentioned_attn = []
    not_mentioned_attn = []
    baseline_attentions = []
    all_files = os.listdir(f"{image_folder}/images")
    if test:        files_to_process = all_files[0:1]
    else:        files_to_process = all_files[:10] if len(all_files) >= 100 else all_files
    for filename in files_to_process:
        image_path = f"{image_folder}/images/{filename}"
        print(f"Processing image: {image_path}")
        plotter = Plotter(image_path)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour
        plotter.set_model(model, processor)

        for pos in ['left', 'right']:

            if pos == 'left':
                plotter.get_outputs(f"The figure is {left_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[left_colour])
                not_mentioned_attn.append(bbox_attentions[right_colour])
                baseline_attentions.append(baseline_attn)
            else:
                plotter.get_outputs(f"The figure is {right_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[right_colour])
                not_mentioned_attn.append(bbox_attentions[left_colour])
                baseline_attentions.append(baseline_attn)

            #baseline_attn = plotter.get_baseline_attention()

        count+=1
    # plot average attention across items
    avg_mentioned_attn = np.mean(np.array(mentioned_attn), axis=0)
    avg_not_mentioned_attn = np.mean(np.array(not_mentioned_attn), axis=0)
    avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)

    return avg_mentioned_attn, avg_not_mentioned_attn, avg_baseline_attn

def get_neg_results(model_name, image_folder, ylim, test=False, save_folder=None, colour=None):

    processor, model = load_model(model_name)
    count = 0
    mentioned_attn = []
    not_mentioned_attn = []
    baseline_attentions = []
    all_files = os.listdir(f"{image_folder}/images")
    if test:        files_to_process = all_files[0:1]
    else:
        files_to_process = all_files[:10] if len(all_files) >= 1000 else all_files
    for filename in files_to_process:
        image_path = f"{image_folder}/images/{filename}"
        print(f"Processing image: {image_path}")
        plotter = Plotter(image_path)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour
        plotter.set_model(model, processor)

        for pos in ['left', 'right']:

            if pos == 'left':
                plotter.get_outputs(f"The figure is not {left_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                baseline_attentions.append(baseline_attn)
                mentioned_attn.append(bbox_attentions[left_colour])
                not_mentioned_attn.append(bbox_attentions[right_colour])
            else:
                plotter.get_outputs(f"The figure is not {right_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[right_colour])
                not_mentioned_attn.append(bbox_attentions[left_colour])
                baseline_attentions.append(baseline_attn)

            #baseline_attn = plotter.get_baseline_attention()

        count+=1
    # plot average attention across items
    avg_mentioned_attn = np.mean(np.array(mentioned_attn), axis=0)
    avg_not_mentioned_attn = np.mean(np.array(not_mentioned_attn), axis=0)
    avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)


    return avg_mentioned_attn, avg_not_mentioned_attn, avg_baseline_attn

def model_worker(model_name, img_folder, ylim, results_queue):
    """
    Handles all GPU logic for a single model and returns plot-ready data.
    """
    # 1. Run your existing extraction logic
    # Make sure these functions return numpy arrays/lists, not Torch tensors!
    pos_m, pos_nm, pos_base = get_pos_results(model_name, img_folder, ylim)
    neg_m, neg_nm, neg_base = get_neg_results(model_name, img_folder, ylim) 
    
    def get_ci(data):
        """Calculate confidence interval for a list of values."""
        if len(data) == 0:
            return np.zeros_like(data)  # Return zeros if no data
        mean = np.mean(data, axis=0)
        sem = np.std(data, axis=0) / np.sqrt(len(data))
        z_score = 1.96  # For 95% confidence
        ci = z_score * sem
        return ci
    # 2. Package data to send back
    # We include everything needed for the plot (means and CIs)
    results = {
        "model_name": model_name,
        "pos": (pos_m, get_ci(pos_m), pos_nm, get_ci(pos_nm), pos_base), 
        "neg": (neg_m, get_ci(neg_m), neg_nm, get_ci(neg_nm), neg_base)

    }
    
    results_queue.put(results)

def plot_results(data_type):
        fig, axs = plt.subplots(2, 3, figsize=(21, 8))
        if data_type == "2_obj":
            full_data_name = "Binary"
        elif data_type == "4_obj":
            full_data_name = "Multary"
        else:
            full_data_name = "What's Up"
        # add title to entire figure
        fig.suptitle(f"{full_data_name}", fontsize=20, fontweight='bold')#, y=1.02
        plt.style.use('tableau-colorblind10') # Good for accessibility
        plt.rcParams['font.family'] = 'sans-serif'
        models = ["paligemma", "internvl", "llava"]

        # Modern color palette
        color_m = '#2c7fb8'  # Focused Blue
        color_nm = '#f03b20' # Focused Red/Orange
        color_base = '#636363' # Neutral Grey

        for i, model_name in enumerate(models):

            max_val = 0
            

            for row, condition in enumerate(["pos", "neg"]):

                pretty_name = f"{model_name.capitalize()} - {'Affirmative' if condition == 'pos' else 'Negated'} " 
                neg_val = "neg_" if condition == "neg" else ""
                mentioned_attn = np.load(f"data_saves_all/{model_name}_{data_type}_{neg_val}mentioned_attn.npy")
                not_mentioned_attn = np.load(f"data_saves_all/{model_name}_{data_type}_{neg_val}non_mentioned_attn.npy")
                print(mentioned_attn.shape)
                if data_type == "4_obj":
                    avg_m = np.mean(mentioned_attn, axis=0)
                    print(avg_m.shape)
                    avg_nm = np.mean(not_mentioned_attn, axis=(0,1))
                    ci_m = np.std(mentioned_attn, axis=(0)) / np.sqrt(len(mentioned_attn)) * 1.96
                    ci_nm = np.std(not_mentioned_attn, axis=(0,1)) / np.sqrt(len(not_mentioned_attn)) * 1.96
                else:   
                
                    avg_m = np.mean(mentioned_attn, axis=0)
                    print(f"data_saves_1000/{model_name}_{data_type}_{neg_val}mentioned_attn.npy")
                    print(avg_m.shape)
                    avg_nm = np.mean(not_mentioned_attn, axis=0)
                    ci_m = np.std(mentioned_attn, axis=0) / np.sqrt(len(mentioned_attn)) * 1.96
                    ci_nm = np.std(not_mentioned_attn, axis=0) / np.sqrt(len(not_mentioned_attn)) * 1.96
                ax = axs[row, i]
                x = list(range(len(avg_m)))
                
                max_val = max(np.max(avg_m + ci_m), np.max(avg_nm + ci_nm), max_val)
                # --- Plotting ---
                # # Baseline first so it stays in the background
                # if plot_baseline:
                #     ax.plot(x, baseline, label='Baseline', color=color_base, linestyle='--', alpha=0.8, linewidth=1.5)
                
                # Not Mentioned
                ax.plot(x, avg_nm, label='Alternative', color=color_nm, linewidth=2.5, zorder=2)
                ax.fill_between(x, avg_nm - ci_nm, avg_nm + ci_nm, color=color_nm, alpha=0.15)
                
                # Mentioned (Z-order higher to stand out)
                ax.plot(x, avg_m, label='Mentioned', color=color_m, linewidth=2.5, zorder=3)
                ax.fill_between(x, avg_m - ci_m, avg_m + ci_m, color=color_m, alpha=0.15)

                # --- Formatting ---
                # Titles only on the top row
                ax.set_title(pretty_name, fontsize=16, fontweight='bold', pad=10)
                
                # Row labels (Y-axis labels)
                ax.set_ylabel("Attention", fontsize=16, fontweight='normal')

                    
                # Clean up Spines (The "Despine" look)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, axis='y', linestyle=':', alpha=0.7)
                
                # Limits and Ticks
                 # Add some headroom above the max value
                ax.tick_params(axis='both', which='major', labelsize=16)
                
                # Legend only on the first plot to avoid clutter
                if i == 0 and row == 0:
                    ax.legend(frameon=False, fontsize=16, loc='upper left')#
            # set global y-limits based on max value across both conditions for this model
            axs[0, i].set_ylim(0, max_val * 1.1)
            axs[1, i].set_ylim(0, max_val * 1.1)

        # Global X-label
        fig.text(0.5, 0.01, 'Layer', ha='center', fontsize=16)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for global x-label
        plt.savefig(f"plots/grouped_plots/{data_type}.png", dpi=300)

if __name__ == "__main__":
    #img_folder = "whatsup"
    # img_folder = "2d_dataset_fixed_positions_1000"
    # models = ['llava', 'internvl', 'paligemma']
    # plot_baseline = True
    # ylims = [0.008, 0.005, 0.02]
    # layers = [32, 36, 26]
    # data_dict = {}

    # # Sequential Multiprocessing: Process one model at a time to save VRAM
    # for model_name, ylim in zip(models, ylims):
    #     print(f"Starting Process for: {model_name}")
        
    #     ctx = mp.get_context('spawn')
    #     q = ctx.Queue()
    #     p = ctx.Process(target=model_worker, args=(model_name, img_folder, ylim, q))
        
    #     p.start()
    #     data_dict[model_name] = q.get() # Grab the data
    #     p.join() # Process ends, GPU memory is fully cleared
        
    #     print(f"Memory cleared for {model_name}")

    # display_names = {
    #     "llava": "LLaVA-1.5",
    #     "internvl": "InternVL3.5",
    #     "paligemma": "PaliGemma2"
    # }

    # plt.style.use('tableau-colorblind10') # Good for accessibility
    # plt.rcParams['font.family'] = 'sans-serif'

    # def plot_results(models, data_dict, layers, display_names, ylims, plot_baseline=True):
    #     fig, axs = plt.subplots(2, 3, figsize=(20, 2), sharex='col')
        
    #     # Modern color palette
    #     color_m = '#2c7fb8'  # Focused Blue
    #     color_nm = '#f03b20' # Focused Red/Orange
    #     color_base = '#636363' # Neutral Grey

    #     for i, model_name in enumerate(models):
    #         res = data_dict[model_name]
    #         x = list(range(layers[i]))
    #         pretty_name = display_names.get(model_name, model_name)

    #         for row, condition in enumerate(["pos", "neg"]):
    #             avg_m, ci_m, avg_nm, ci_nm, baseline = res[condition]
    #             ax = axs[row, i]
                
    #             # --- Plotting ---
    #             # Baseline first so it stays in the background
    #             if plot_baseline:
    #                 ax.plot(x, baseline, label='Baseline', color=color_base, linestyle='--', alpha=0.8, linewidth=1.5)
                
    #             # Not Mentioned
    #             ax.plot(x, avg_nm, label='Not Mentioned', color=color_nm, linewidth=2.5, zorder=2)
    #             ax.fill_between(x, avg_nm - ci_nm, avg_nm + ci_nm, color=color_nm, alpha=0.15)
                
    #             # Mentioned (Z-order higher to stand out)
    #             ax.plot(x, avg_m, label='Mentioned', color=color_m, linewidth=2.5, zorder=3)
    #             ax.fill_between(x, avg_m - ci_m, avg_m + ci_m, color=color_m, alpha=0.15)

    #             # --- Formatting ---
    #             # Titles only on the top row
    #             if row == 0:
    #                 ax.set_title(pretty_name, fontsize=18, fontweight='bold', pad=20)
                
    #             # Row labels (Y-axis labels)
    #             if i == 0:
    #                 label = "Positive Questions" if condition == "pos" else "Negative Questions"
    #                 ax.set_ylabel(f"{label}\nAttention Score", fontsize=14, fontweight='semibold')
                
    #             # Clean up Spines (The "Despine" look)
    #             ax.spines['top'].set_visible(False)
    #             ax.spines['right'].set_visible(False)
    #             ax.grid(True, axis='y', linestyle=':', alpha=0.7)
                
    #             # Limits and Ticks
    #             ax.set_ylim(0, ylims[i])
    #             ax.tick_params(axis='both', which='major', labelsize=12)
                
    #             # Legend only on the first plot to avoid clutter
    #             if i == 0 and row == 0:
    #                 ax.legend(frameon=False, fontsize=12, loc='upper left')

    #     # Global X-label
    #     fig.text(0.5, 0.01, 'Transformer Layer Index', ha='center', fontsize=16, fontweight='semibold')
        
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for global x-label
    #     plt.savefig("plots/grouped_plots/2_object.png", dpi=300)

    plot_results("4_obj")