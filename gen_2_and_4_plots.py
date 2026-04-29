from plotter import Plotter
import torch 
import os 
from transformers import AutoProcessor, AutoModelForImageTextToText, InternVLForConditionalGeneration, LlavaForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import gc
from matplotlib import rcParams
import seaborn as sns
import scipy

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


def grab_2_data(model_name, image_dir, save_dir):

    processor, model = load_model(model_name)
    for neg in [False, True]:
        mentioned_attn = []
        non_mentioned_attn = []
        for image_path in os.listdir(image_dir+"/images/"):
            plotter = Plotter(f"{image_dir}/images/{image_path}")
            left_colour = plotter.get_left_shapes()[0].colour
            right_colour = plotter.get_right_shapes()[0].colour

            # get attention for both left and right being mentioned and average
            for pos in ["left", "right"]:

                    if pos == "left":
                        mentioned_colour = left_colour
                        unmentioned_colour = right_colour
                    else:
                        mentioned_colour = right_colour
                        unmentioned_colour = left_colour

                    neg_val = "not " if neg else ""
                    plotter.set_model(model, processor)
                    plotter.get_outputs(f"The figure is {neg_val}{mentioned_colour}")
                    print(f"The figure is {neg_val}{mentioned_colour}")
                    
                    bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                    mentioned_attn.append(bbox_attentions[mentioned_colour])
                    non_mentioned_attn.append(bbox_attentions[unmentioned_colour])

        save_neg = "neg_" if neg else ""
        # save data as numpy arrays
        np.save(save_dir + f"/{model_name}_whatsup_{save_neg}mentioned_attn.npy", np.array(mentioned_attn))
        np.save(save_dir + f"/{model_name}_whatsup_{save_neg}non_mentioned_attn.npy", np.array(non_mentioned_attn))


                # 5. Cleanup
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Finished plotting for {model_name}")


def grab_4_data(model_name, image_dir, save_dir):
    
    processor, model = load_model(model_name)

    

    for neg in [False, True]:
        mentioned_attn = []
        non_mentioned_attn = []
        for image_path in os.listdir(image_dir+"/images/")[0:100]:
            plotter = Plotter(f"{image_dir}/images/{image_path}")

            for pos in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:

                    top_left_colour = plotter.get_shape_by_position('top_left').colour
                    top_right_colour = plotter.get_shape_by_position('top_right').colour
                    bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
                    bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour
                    plotter.set_model(model, processor)
                    
                    if pos == 'top_left' or pos == 'neg_top_left':
                        target_colour = top_left_colour
                    elif pos == 'top_right' or pos == 'neg_top_right':
                        target_colour = top_right_colour
                    elif pos == 'bottom_left' or pos == 'neg_bottom_left':
                        target_colour = bottom_left_colour
                    elif pos == 'bottom_right' or pos == 'neg_bottom_right':
                        target_colour = bottom_right_colour

                    neg_val = "not " if neg else ""
                    plotter.get_outputs(f"The figure is {neg_val}{target_colour}")
                    print(f"The figure is {neg_val}{target_colour}")

                    bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                    if pos == 'top_left':
                        mentioned_attn.append(bbox_attentions[top_left_colour])
                        non_mentioned_attn.append([bbox_attentions[top_right_colour], bbox_attentions[bottom_left_colour], bbox_attentions[bottom_right_colour]])
                    elif pos == 'top_right':
                        mentioned_attn.append(bbox_attentions[top_right_colour])
                        non_mentioned_attn.append([bbox_attentions[top_left_colour], bbox_attentions[bottom_left_colour], bbox_attentions[bottom_right_colour]])
                    elif pos == 'bottom_left':  
                        mentioned_attn.append(bbox_attentions[bottom_left_colour])
                        non_mentioned_attn.append([bbox_attentions[top_left_colour], bbox_attentions[top_right_colour], bbox_attentions[bottom_right_colour]])
                    elif pos == 'bottom_right':
                        mentioned_attn.append(bbox_attentions[bottom_right_colour])
                        non_mentioned_attn.append([bbox_attentions[top_left_colour], bbox_attentions[top_right_colour], bbox_attentions[bottom_left_colour]])

                
        save_neg = "neg_" if neg else ""
        # save data as numpy arrays
        np.save(save_dir + f"/{model_name}_4_obj_{save_neg}mentioned_attn.npy", np.array(mentioned_attn))
        np.save(save_dir + f"/{model_name}_4_obj_{save_neg}non_mentioned_attn.npy", np.array(non_mentioned_attn))

def generate_graphs(save_dir):

    for model_name in ['llava', 'internvl', 'paligemma']:

        # plot 2 plots in a row for 2 object dataset and 4 object dataset
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        all_data_max = 0
        for dataset_type in ["2_obj", "4_obj"]:
            # plot mentioned positive, mentioned negative, non-mentioned positive, non-mentioned negative on same graph with error bars
            mentioned_pos = np.load(save_dir + f"/{model_name}_{dataset_type}_mentioned_attn.npy")
            mentioned_pos = np.mean(mentioned_pos, axis=0)
            mentioned_neg = np.load(save_dir + f"/{model_name}_{dataset_type}_neg_mentioned_attn.npy")
            mentioned_neg = np.mean(mentioned_neg, axis=0)
            non_mentioned_pos = np.load(save_dir + f"/{model_name}_{dataset_type}_non_mentioned_attn.npy")
            non_mentioned_neg = np.load(save_dir + f"/{model_name}_{dataset_type}_neg_non_mentioned_attn.npy")
            if dataset_type == "2_obj":
                non_mentioned_pos = np.mean(non_mentioned_pos, axis=0)
                non_mentioned_neg = np.mean(non_mentioned_neg, axis=0)
            else:
                # nonmentioned pos is a list of lists, taking mean axis=0 gives shape (3,32) we want shape (1, 32)
                non_mentioned_pos = np.mean(non_mentioned_pos, axis=(0,1))
                non_mentioned_neg = np.mean(non_mentioned_neg, axis=(0,1))

            current_max = max(mentioned_pos.max(), mentioned_neg.max(), 
                        non_mentioned_pos.max(), non_mentioned_neg.max())
            all_data_max = max(all_data_max, current_max)
            layers = list(range(len(mentioned_pos)))
            ax = axes[0] if dataset_type == "2_obj" else axes[1]
            ax.plot(layers, mentioned_pos, label='Mentioned Positive', color='blue')
            ax.plot(layers, mentioned_neg, label='Mentioned Negative', color='blue', linestyle='--')
            ax.plot(layers, non_mentioned_pos, label='Non-Mentioned Positive', color='red')
            ax.plot(layers, non_mentioned_neg, label='Non-Mentioned Negative', color='red', linestyle='--')
            ax.set_xlabel('Layer')
            dataset_type_formatted = "2 Object" if dataset_type == "2_obj" else "4 Object"
            ax.set_title(f"{dataset_type_formatted} Dataset")
            # fix y axis limit based on max of all 

        for ax in axes:
            ax.set_ylim(0, all_data_max * 1.1)

        handles, labels = axes[0].get_legend_handles_labels()
        lgd = figure.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        plt.savefig(f"plots/pos_neg_combined/{model_name}.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()


def generate_bar_graphs(save_dir):

    model_names = ['llava', 'internvl', 'paligemma']
    dataset_types = ["2_obj", "4_obj"]
    # Colors: Using a softer, modern palette
    color_mentioned = '#2c7fb8'     # Professional Blue
    color_non_mentioned = '#f03b20' # Soft Crimson/Orange
    bar_width = 0.18                # Slightly wider bars
    dataset_gap = 0.05              # Much smaller gap between 2_obj and 4_obj
    group_gap = 0.6                 # Clearer gap between models

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    all_means = []
    dataset_labels = ["2 Object", "4 Object"]
    dataset_tick_positions = []
    dataset_tick_labels = []

    for i, model_name in enumerate(model_names):
        for j, dataset_type in enumerate(dataset_types):
            # ... [Your existing np.load and np.mean logic here] ...
            mentioned_pos = np.load(save_dir + f"/{model_name}_{dataset_type}_mentioned_attn.npy")
            mentioned_pos = np.mean(mentioned_pos, axis=0)
            mentioned_neg = np.load(save_dir + f"/{model_name}_{dataset_type}_neg_mentioned_attn.npy")
            mentioned_neg = np.mean(mentioned_neg, axis=0)
            non_mentioned_pos = np.load(save_dir + f"/{model_name}_{dataset_type}_non_mentioned_attn.npy")
            non_mentioned_neg = np.load(save_dir + f"/{model_name}_{dataset_type}_neg_non_mentioned_attn.npy")
            if dataset_type == "2_obj":
                non_mentioned_pos = np.mean(non_mentioned_pos, axis=0)
                non_mentioned_neg = np.mean(non_mentioned_neg, axis=0)
            else:
                # nonmentioned pos is a list of lists, taking mean axis=0 gives shape (3,32) we want shape (1, 32)
                non_mentioned_pos = np.mean(non_mentioned_pos, axis=(0,1))
                non_mentioned_neg = np.mean(non_mentioned_neg, axis=(0,1))
            
        # (Model Index * Total Group Width) + (Dataset Index * Dataset Group Width)
            base_x = i * (8 * bar_width + dataset_gap + group_gap) + j * (4 * bar_width + dataset_gap)
            center = base_x + 1.5 * bar_width
            dataset_tick_positions.append(center)
            label = f"{dataset_labels[j]}"
            dataset_tick_labels.append(label)
            vals = [mentioned_pos.mean(), mentioned_neg.mean(), 
                    non_mentioned_pos.mean(), non_mentioned_neg.mean()]
            all_means.extend(vals)
            
            # Plotting the 4 bars for this dataset
            # Bar 1: Mentioned Pos
            ax.bar(base_x + 0*bar_width, vals[0], width=bar_width, color=color_mentioned, 
                edgecolor='white', linewidth=0.5, label='Mentioned Positive' if i==0 and j==0 else "")
            # Bar 2: Mentioned Neg (Hatched)
            ax.bar(base_x + 1*bar_width, vals[1], width=bar_width, color=color_mentioned, alpha=0.7,
                hatch='////', edgecolor='white', linewidth=0.5, label='Mentioned Negative' if i==0 and j==0 else "")
            # Bar 3: Non-Mentioned Pos
            ax.bar(base_x + 2*bar_width, vals[2], width=bar_width, color=color_non_mentioned, 
                edgecolor='white', linewidth=0.5, label='Non-Mentioned Positive' if i==0 and j==0 else "")
            # Bar 4: Non-Mentioned Neg (Hatched)
            ax.bar(base_x + 3*bar_width, vals[3], width=bar_width, color=color_non_mentioned, alpha=0.7,
                hatch='////', edgecolor='white', linewidth=0.5, label='Non-Mentioned Negative' if i==0 and j==0 else "")

    # --- Styling ---

    # Clean up the spines (borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True) # Grid goes behind bars

    # Center the model names
    tick_centers = []
    for i in range(len(model_names)):
        start = i * (8 * bar_width + dataset_gap + group_gap)
        end = start + (8 * bar_width + dataset_gap)
        tick_centers.append((start + end) / 2 - (bar_width/2))

    ax.set_xticks(tick_centers)
    model_names_formatted = [name.upper() for name in model_names]
    ax.set_xticklabels(model_names_formatted, fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Attention', fontsize=10)
    ax.set_title('Object Attention: Mentioned vs. Non-Mentioned', fontsize=13, pad=20)
    ax2 = ax.secondary_xaxis('bottom')
    ax2.set_xticks(dataset_tick_positions)
    ax2.set_xticklabels(dataset_tick_labels, fontsize=12)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(length=0)  # Hide tick marks

    # Shift the model name ticks up slightly to make room
    ax.tick_params(axis='x', pad=18)
    # Legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                    ncol=4, frameon=False, fontsize=10)

    plt.savefig("plots/pos_neg_combined/all_models_bar_refined.png", 
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def generate_bar_graphs_all_models(save_dir):


    model_names = ['llava', 'internvl', 'paligemma']
    dataset_types = ["2_obj", "4_obj", "whatsup"]
    dataset_labels = ["2 Object", "4 Object", "What's Up"]
    
    color_mentioned = '#2c7fb8'     
    color_non_mentioned = '#f03b20' 
    bar_width = 0.18            
    group_spacing = 0.2  

    # 1. sharey=False allows each model to have its own vertical scale
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi=120, sharey=False)
    fig.subplots_adjust(wspace=0.3) # Increased width space for y-axis labels

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        dataset_tick_positions = []

        for j, dataset_type in enumerate(dataset_types):
            # --- Data Loading Logic ---
            m_pos = np.mean(np.load(f"{save_dir}/{model_name}_{dataset_type}_mentioned_attn.npy"), axis=0)
            m_neg = np.mean(np.load(f"{save_dir}/{model_name}_{dataset_type}_neg_mentioned_attn.npy"), axis=0)
            nm_pos = np.load(f"{save_dir}/{model_name}_{dataset_type}_non_mentioned_attn.npy")
            nm_neg = np.load(f"{save_dir}/{model_name}_{dataset_type}_neg_non_mentioned_attn.npy")

            if dataset_type == "2_obj" or dataset_type == "whatsup":
                nm_pos = np.mean(nm_pos, axis=0)
                nm_neg = np.mean(nm_neg, axis=0)
            else:
                nm_pos = np.mean(nm_pos, axis=(0, 1))
                nm_neg = np.mean(nm_neg, axis=(0, 1))

            vals = [m_pos.mean(), m_neg.mean(), nm_pos.mean(), nm_neg.mean()] 
            
            base_x = j * (4 * bar_width + group_spacing)
            center = base_x + (1.5 * bar_width)
            dataset_tick_positions.append(center)

            # --- Plotting ---
            ax.bar(base_x + 0*bar_width, vals[0], width=bar_width, color=color_mentioned, 
                        edgecolor='white', linewidth=0.5, label='Mentioned Positive' if (i==0 and j==0) else "")
            ax.bar(base_x + 1*bar_width, vals[1], width=bar_width, color=color_mentioned, alpha=0.7,    
                        hatch='////', edgecolor='white', linewidth=0.5, label='Mentioned Negated' if (i==0 and j==0) else "")     
            ax.bar(base_x + 2*bar_width, vals[2], width=bar_width, color=color_non_mentioned, 
                        edgecolor='white', linewidth=0.5, label='Not Mentioned Positive' if (i==0 and j==0) else "")   
            ax.bar(base_x + 3*bar_width, vals[3], width=bar_width, color=color_non_mentioned, alpha=0.7,
                        hatch='////', edgecolor='white', linewidth=0.5, label='Not Mentioned Negated' if (i==0 and j==0) else "") 

        # --- Subplot Styling ---
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        
        # 2. Add y-labels and ensure ticks are visible for every graph
        ax.set_ylabel('Mean Attention', fontsize=12)
        ax.tick_params(axis='y', labelleft=True) 
        
        ax.set_xticks(dataset_tick_positions)
        ax.set_xticklabels(dataset_labels, fontsize=12)
        ax.set_xlim(dataset_tick_positions[0] - 0.5, dataset_tick_positions[-1] + 0.5)
        
        ax.set_title(model_name.upper(), fontsize=14, fontweight='bold', pad=15)

    # --- Global Legend ---
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/pos_neg_combined/triple_model_independent_scales.png", bbox_inches='tight')
    plt.show()

def generate_bar_graphs_model(model_name, save_dir):
    dataset_types = ["2_obj", "4_obj", "whatsup"]
    color_mentioned = '#2c7fb8'     
    color_non_mentioned = '#f03b20' 
    bar_width = 0.18            


    # Increase this value to add more whitespace between "2 Object" and "4 Object"
    group_spacing = 0.15  

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120) # Slightly taller for better proportions
    dataset_labels = ["2 Object", "4 Object", "What's Up"]
    dataset_tick_positions = []

    for j, dataset_type in enumerate(dataset_types):
        mentioned_pos = np.load(save_dir + f"/{model_name}_{dataset_type}_mentioned_attn.npy")
        mentioned_pos = np.mean(mentioned_pos, axis=0)
        mentioned_neg = np.load(save_dir + f"/{model_name}_{dataset_type}_neg_mentioned_attn.npy")
        mentioned_neg = np.mean(mentioned_neg, axis=0)
        non_mentioned_pos = np.load(save_dir + f"/{model_name}_{dataset_type}_non_mentioned_attn.npy")
        non_mentioned_neg = np.load(save_dir + f"/{model_name}_{dataset_type}_neg_non_mentioned_attn.npy")
        if dataset_type == "2_obj" or dataset_type == "whatsup":
            non_mentioned_pos = np.mean(non_mentioned_pos, axis=0)
            non_mentioned_neg = np.mean(non_mentioned_neg, axis=0)
        else:
            # nonmentioned pos is a list of lists, taking mean axis=0 gives shape (3,32) we want shape (1, 32)
            non_mentioned_pos = np.mean(non_mentioned_pos, axis=(0,1))
            non_mentioned_neg = np.mean(non_mentioned_neg, axis=(0,1))
            

        # ... [Your data loading logic remains the same] ...
        # (Assuming data processing for mentioned_pos, mentioned_neg, etc. happens here)

        vals = [mentioned_pos.mean(), mentioned_neg.mean(), 
                non_mentioned_pos.mean(), non_mentioned_neg.mean()] 
        
        base_x = j * (4 * bar_width + group_spacing)
        
        # Center the label under the 4 bars
        center = base_x + (1.5 * bar_width)
        dataset_tick_positions.append(center)

        # --- Plotting ---
        # Labels only added for the first iteration (j==0) to avoid legend duplicates
        ax.bar(base_x + 0*bar_width, vals[0], width=bar_width, color=color_mentioned, 
               edgecolor='white', linewidth=0.5, label='Mentioned Positive' if j==0 else "")
        ax.bar(base_x + 1*bar_width, vals[1], width=bar_width, color=color_mentioned, alpha=0.7,    
               hatch='////', edgecolor='white', linewidth=0.5, label='Mentioned Negated' if j==0 else "")     
        ax.bar(base_x + 2*bar_width, vals[2], width=bar_width, color=color_non_mentioned, 
               edgecolor='white', linewidth=0.5, label='Not Mentioned Positive' if j==0 else "")   
        ax.bar(base_x + 3*bar_width, vals[3], width=bar_width, color=color_non_mentioned, alpha=0.7,
               hatch='////', edgecolor='white', linewidth=0.5, label='Not Mentioned Negated' if j==0 else "") 
        
    # --- Styling & Legend Adjustment ---   
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    
    # Set the ticks and labels for all 3 datasets
    ax.set_xticks(dataset_tick_positions)
    ax.set_xticklabels(dataset_labels, fontsize=20)
    
    # Dynamic X-Limit to provide breathing room on both sides
    ax.set_xlim(dataset_tick_positions[0] - 0.6, dataset_tick_positions[-1] + 0.6)
    
    ax.set_ylabel('Mean Attention', fontsize=20)
    ax.set_title(f'{model_name.upper()}', fontsize=20, pad=20)  
    
    # FIX: ncol=4 puts them in one line. 
    # bbox_to_anchor y-value adjusted to -0.2 to clear the dataset labels.
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=20)
    
    plt.savefig(f"plots/pos_neg_combined/{model_name}_bar_refined.png", bbox_inches='tight')
    plt.show()



def generate_graph_3_datasets(model_name, save_dir):

    ## load all data in 
    mentioned_pos_2 = np.load(save_dir + f"/{model_name}_2_obj_mentioned_attn.npy")
    mentioned_neg_2 = np.load(save_dir + f"/{model_name}_2_obj_neg_mentioned_attn.npy")
    non_mentioned_pos_2 = np.load(save_dir + f"/{model_name}_2_obj_non_mentioned_attn.npy")
    non_mentioned_neg_2 = np.load(save_dir + f"/{model_name}_2_obj_neg_non_mentioned_attn.npy") 
    mentioned_pos_4 = np.load(save_dir + f"/{model_name}_4_obj_mentioned_attn.npy")
    mentioned_neg_4 = np.load(save_dir + f"/{model_name}_4_obj_neg_mentioned_attn.npy")
    non_mentioned_pos_4 = np.load(save_dir + f"/{model_name}_4_obj_non_mentioned_attn.npy")
    non_mentioned_neg_4 = np.load(save_dir + f"/{model_name}_4_obj_neg_non_mentioned_attn.npy") 
    mentioned_pos_whatsup = np.load(save_dir + f"/{model_name}_whatsup_mentioned_attn.npy")
    mentioned_neg_whatsup = np.load(save_dir + f"/{model_name}_whatsup_neg_mentioned_attn.npy")
    non_mentioned_pos_whatsup = np.load(save_dir + f"/{model_name}_whatsup_non_mentioned_attn.npy")
    non_mentioned_neg_whatsup = np.load(save_dir + f"/{model_name}_whatsup_neg_non_mentioned_attn.npy")
    # calculate averages
    mentioned_pos_2 = np.mean(mentioned_pos_2, axis=0)
    mentioned_neg_2 = np.mean(mentioned_neg_2, axis=0)
    non_mentioned_pos_2 = np.mean(non_mentioned_pos_2, axis=0)
    non_mentioned_neg_2 = np.mean(non_mentioned_neg_2, axis=0)
    mentioned_pos_4 = np.mean(mentioned_pos_4, axis=0)
    mentioned_neg_4 = np.mean(mentioned_neg_4, axis=0)
    non_mentioned_pos_4 = np.mean(non_mentioned_pos_4, axis=(0,1))
    non_mentioned_neg_4 = np.mean(non_mentioned_neg_4, axis=(0,1))
    mentioned_pos_whatsup = np.mean(mentioned_pos_whatsup, axis=0)
    mentioned_neg_whatsup = np.mean(mentioned_neg_whatsup, axis=0)
    non_mentioned_pos_whatsup = np.mean(non_mentioned_pos_whatsup, axis=0)
    non_mentioned_neg_whatsup = np.mean(non_mentioned_neg_whatsup, axis=0)  

    datasets = {
        '2 Objects': {
            'Mentioned Pos':     mentioned_pos_2,
            'Mentioned Neg':     mentioned_neg_2,
            'Non-Mentioned Pos': non_mentioned_pos_2,
            'Non-Mentioned Neg': non_mentioned_neg_2,
        },
        '4 Objects': {
            'Mentioned Pos':     mentioned_pos_4,
            'Mentioned Neg':     mentioned_neg_4,
            'Non-Mentioned Pos': non_mentioned_pos_4,
            'Non-Mentioned Neg': non_mentioned_neg_4,
        },
        "What's Up": {
            'Mentioned Pos':     mentioned_pos_whatsup,
            'Mentioned Neg':     mentioned_neg_whatsup,
            'Non-Mentioned Pos': non_mentioned_pos_whatsup,
            'Non-Mentioned Neg': non_mentioned_neg_whatsup,
        },
    }

    styles = {
        'Mentioned Pos':     dict(color='#2c7fb8', linestyle='-',  linewidth=2,   label='Mentioned (Pos)'),
        'Mentioned Neg':     dict(color='#f03b20', linestyle='-', linewidth=2,   label='Mentioned (Neg)'),
        'Non-Mentioned Pos': dict(color='#2c7fb8', linestyle='--',  linewidth=2,   label='Non-Mentioned (Pos)'),
        'Non-Mentioned Neg': dict(color='#f03b20', linestyle='--', linewidth=2,   label='Non-Mentioned (Neg)'),
        }

    fig, axes = plt.subplots(1, 3, figsize=(8, 3), dpi=120, sharey=True)
    fig.suptitle(f'Attention Across Datasets — {model_name.upper()}',
                fontsize=14, fontweight='bold', y=1.02)

    for ax, (dataset_name, series) in zip(axes, datasets.items()):
        for key, values in series.items():
            ax.plot(values, **styles[key])
        
        # Shade the gap between mentioned and non-mentioned means
        mean_mentioned     = (series['Mentioned Pos']     + series['Mentioned Neg'])     / 2
        mean_non_mentioned = (series['Non-Mentioned Pos'] + series['Non-Mentioned Neg']) / 2

        ax.set_title(dataset_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('Mean Attention')

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
            bbox_to_anchor=(0.5, -0.08), fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"plots/pos_neg_combined/{model_name}_all_datasets.png",
                bbox_inches='tight')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def gen_multary_binary_plots(model_name, save_dir):
    # Load data
    mentioned_pos_2 = np.load(save_dir + f"/{model_name}_2_obj_mentioned_attn.npy")
    mentioned_neg_2 = np.load(save_dir + f"/{model_name}_2_obj_neg_mentioned_attn.npy")
    non_mentioned_pos_2 = np.load(save_dir + f"/{model_name}_2_obj_non_mentioned_attn.npy")
    non_mentioned_neg_2 = np.load(save_dir + f"/{model_name}_2_obj_neg_non_mentioned_attn.npy")
    mentioned_pos_4 = np.load(save_dir + f"/{model_name}_4_obj_mentioned_attn.npy")
    mentioned_neg_4 = np.load(save_dir + f"/{model_name}_4_obj_neg_mentioned_attn.npy")
    non_mentioned_pos_4 = np.load(save_dir + f"/{model_name}_4_obj_non_mentioned_attn.npy")
    non_mentioned_neg_4 = np.load(save_dir + f"/{model_name}_4_obj_neg_non_mentioned_attn.npy")

    # Compute mean ± SE
    def mean_se(arr, axis):
        n = arr.shape[0] if isinstance(axis, tuple) else arr.shape[axis]
        m  = np.mean(arr, axis=axis)
        se = np.std(arr,  axis=axis) / np.sqrt(n)
        return m, se

    mp2, mp2_se = mean_se(mentioned_pos_2,     axis=0)
    mn2, mn2_se = mean_se(mentioned_neg_2,     axis=0)
    np2, np2_se = mean_se(non_mentioned_pos_2, axis=0)
    nn2, nn2_se = mean_se(non_mentioned_neg_2, axis=0)
    mp4, mp4_se = mean_se(mentioned_pos_4,     axis=0)
    mn4, mn4_se = mean_se(mentioned_neg_4,     axis=0)
    np4, np4_se = mean_se(non_mentioned_pos_4, axis=(0, 1))
    nn4, nn4_se = mean_se(non_mentioned_neg_4, axis=(0, 1))

    # Style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    C2 = "#2563EB"   # blue  — 2 objects
    C4 = "#DC2626"   # red   — 4 objects
    BG = "#F8F9FB"
    ALPHA = 0.12

    fig, (ax_m, ax_nm) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.patch.set_facecolor(BG)

    layers = np.arange(len(mp2))

    panels = [
        (ax_m,  "Mentioned",     mp2, mp2_se, mn2, mn2_se, mp4, mp4_se, mn4, mn4_se),
        (ax_nm, "Non-Mentioned", np2, np2_se, nn2, nn2_se, np4, np4_se, nn4, nn4_se),
    ]

    for ax, title, pos2, pos2_se, neg2, neg2_se, pos4, pos4_se, neg4, neg4_se in panels:
        ax.set_facecolor(BG)

        # 2-object lines (blue)
        ax.plot(layers, pos2, color=C2, linestyle="-",  linewidth=2.2, zorder=3)
        ax.fill_between(layers, pos2 - pos2_se, pos2 + pos2_se, color=C2, alpha=ALPHA)

        ax.plot(layers, neg2, color=C2, linestyle="--", linewidth=1.8, zorder=3)
        ax.fill_between(layers, neg2 - neg2_se, neg2 + neg2_se, color=C2, alpha=ALPHA)

        # 4-object lines (red)
        ax.plot(layers, pos4, color=C4, linestyle="-",  linewidth=2.2, zorder=3)
        ax.fill_between(layers, pos4 - pos4_se, pos4 + pos4_se, color=C4, alpha=ALPHA)

        ax.plot(layers, neg4, color=C4, linestyle="--", linewidth=1.8, zorder=3)
        ax.fill_between(layers, neg4 - neg4_se, neg4 + neg4_se, color=C4, alpha=ALPHA)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Layer", fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.4, color="#9CA3AF")
        ax.grid(axis="x", linestyle=":", alpha=0.2, color="#9CA3AF")
        ax.tick_params(labelsize=10)

    ax_m.set_ylabel("Mean Attention", fontsize=11)

    # Shared legend: color = object count, style = polarity
    color_handles = [
        mpatches.Patch(color=C2, label="2 Objects"),
        mpatches.Patch(color=C4, label="4 Objects"),
    ]
    style_handles = [
        plt.Line2D([0], [0], color="gray", linewidth=1.8, linestyle="-",  label="Positive"),
        plt.Line2D([0], [0], color="gray", linewidth=1.8, linestyle="--", label="Negative"),
    ]
    fig.legend(
        handles=color_handles + style_handles,
        loc="lower center",
        ncol=4,
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        edgecolor="#D1D5DB",
        bbox_to_anchor=(0.5, -0.07),
    )

    fig.suptitle(
        f"Attention by Mention Status  ·  {model_name.upper()}",
        fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(
        f"plots/pos_neg_combined/{model_name}_3_datasets.png",
        dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.show()


def generate_scatter_plots_all_models(save_dir):

    model_names = ['llava', 'internvl', 'paligemma']
    dataset_types = ["2_obj", "4_obj", "whatsup"]
    dataset_labels = ["2 Object", "4 Object", "What's Up"]

    color_mentioned = '#2c7fb8'
    color_non_mentioned = '#f03b20'

    conditions = [
        ('Mentioned Positive',     color_mentioned,     'o', 1.0,  0),
        ('Mentioned Negated',      color_mentioned,     's', 0.5,  1),
        ('Not Mentioned Positive', color_non_mentioned, 'o', 1.0,  2),
        ('Not Mentioned Negated',  color_non_mentioned, 's', 0.5,  3),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi=120, sharey=False)
    fig.subplots_adjust(wspace=0.35)

    x_positions = np.arange(len(dataset_types))  # [0, 1, 2]
    offsets = np.linspace(-0.27, 0.27, 4)         # 4 conditions spread around each x tick

    for i, model_name in enumerate(model_names):
        ax = axes[i]

        all_arrays = []
        for dataset_type in dataset_types:
            m_pos  = np.mean(np.load(f"{save_dir}/{model_name}_{dataset_type}_mentioned_attn.npy"), axis=0)
            m_neg  = np.mean(np.load(f"{save_dir}/{model_name}_{dataset_type}_neg_mentioned_attn.npy"), axis=0)
            nm_pos = np.load(f"{save_dir}/{model_name}_{dataset_type}_non_mentioned_attn.npy")
            nm_neg = np.load(f"{save_dir}/{model_name}_{dataset_type}_neg_non_mentioned_attn.npy")

            if dataset_type in ("2_obj", "whatsup"):
                nm_pos = np.mean(nm_pos, axis=0)
                nm_neg = np.mean(nm_neg, axis=0)
            else:
                nm_pos = np.mean(nm_pos, axis=(0, 1))
                nm_neg = np.mean(nm_neg, axis=(0, 1))

            # Each array is now 1-D (flattened per-sample values)
            all_arrays.append([m_pos.flatten(), m_neg.flatten(),
                                nm_pos.flatten(), nm_neg.flatten()])

        for k, (label, color, marker, alpha, cond_idx) in enumerate(conditions):
            means, ci_lows, ci_highs = [], [], []
            for j in range(len(dataset_types)):
                data = all_arrays[j][cond_idx]
                n    = len(data)
                mean = data.mean()
                se   = data.std(ddof=1) / np.sqrt(n)
                # 95% CI using t-distribution
                t_val = scipy.stats.t.ppf(0.975, df=n - 1)
                ci    = t_val * se
                means.append(mean)
                ci_lows.append(ci)
                ci_highs.append(ci)

            x = x_positions + offsets[k]
            ax.errorbar(
                x, means,
                yerr=[ci_lows, ci_highs],
                fmt=marker,
                color=color,
                alpha=alpha,
                markersize=10,
                capsize=6,
                capthick=1.2,
                elinewidth=1.2,
                linewidth=0,          # no connecting line between datasets
                label=label if i == 0 else "",
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel('Mean Attention', fontsize=12)
        ax.tick_params(axis='y', labelleft=True)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(dataset_labels, fontsize=12)
        ax.set_xlim(-0.6, len(dataset_types) - 0.4)
        ax.set_title(model_name.upper(), fontsize=12, fontweight='bold', pad=15)

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots/pos_neg_combined/triple_model_scatter_ci.png", bbox_inches='tight')
    plt.show()

def generate_bar_graphs_all_models_ci(save_dir):

    model_names = ['paligemma', 'internvl', 'llava']
    dataset_types = ["2_obj", "4_obj", "whatsup"]
    dataset_labels = ["Bin.", "Mul.", "What's Up"]
    
    color_mentioned = '#2c7fb8'     
    color_non_mentioned = '#f03b20' 
    bar_width = 0.18            
    group_spacing = 0.2
    CI_Z = 1.645 #1.96  # 95% confidence interval

    def mean_and_ci(arr):
        """Return (mean, half-width CI) for a 1D array."""
        n = len(arr)
        se = np.std(arr, ddof=1) / np.sqrt(n)
        return arr.mean(), CI_Z * se

    fig, axes = plt.subplots(1, 3, figsize=(11, 3), dpi=120, sharey=False)
    fig.subplots_adjust(wspace=0.3)

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        dataset_tick_positions = []

        for j, dataset_type in enumerate(dataset_types):
            # --- Data Loading ---
            if dataset_type =="4_obj":
                save_dir ="data_saves_all"
            m_pos = np.mean(np.load(f"{save_dir}/{model_name}_{dataset_type}_mentioned_attn.npy"), axis=0)
            m_neg = np.mean(np.load(f"{save_dir}/{model_name}_{dataset_type}_neg_mentioned_attn.npy"), axis=0)
            nm_pos = np.load(f"{save_dir}/{model_name}_{dataset_type}_non_mentioned_attn.npy")
            nm_neg = np.load(f"{save_dir}/{model_name}_{dataset_type}_neg_non_mentioned_attn.npy")

            if dataset_type == "2_obj" or dataset_type == "whatsup":
                nm_pos = np.mean(nm_pos, axis=0)
                nm_neg = np.mean(nm_neg, axis=0)
            else:
                nm_pos = np.mean(nm_pos, axis=(0, 1))
                nm_neg = np.mean(nm_neg, axis=(0, 1))

            # --- Compute means + CIs from the final 1D arrays ---
            means_cis = [mean_and_ci(arr.flatten()) for arr in [m_pos, m_neg, nm_pos, nm_neg]]
            vals  = [mc[0] for mc in means_cis]
            yerrs = [mc[1] for mc in means_cis]
            
            base_x = j * (4 * bar_width + group_spacing)
            center = base_x + (1.5 * bar_width)
            dataset_tick_positions.append(center)

            # --- Plotting ---
            bar_configs = [
                (0, color_mentioned,     1.0,  ''),
                (1, color_mentioned,     0.7, '////'),
                (2, color_non_mentioned, 1.0,  ''),
                (3, color_non_mentioned, 0.7, '////'),
            ]
            labels = [
                'Mentioned', 'Mentioned Negated',
                'Alternative', 'Alternative Negated'
            ]
            for k, (offset, color, alpha, hatch) in enumerate(bar_configs):
                ax.bar(
                    base_x + offset * bar_width, vals[k],
                    width=bar_width, color=color, alpha=alpha,
                    hatch=hatch, edgecolor='white', linewidth=0.5,
                    label=labels[k] if (i == 0 and j == 0) else ""
                )
                ax.errorbar(
                    base_x + offset * bar_width, vals[k],
                    yerr=yerrs[k],
                    fmt='none',           # no marker, just the error bar
                    ecolor='#333333',
                    elinewidth=1.0,
                    capsize=3,
                    capthick=1.0,
                    zorder=5              # draw on top of bars
                )

        # --- Subplot Styling ---
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel('Mean Attention', fontsize=12)
        ax.tick_params(axis='y', labelleft=True) 
        ax.set_xticks(dataset_tick_positions)
        ax.set_xticklabels(dataset_labels, fontsize=12)
        ax.set_xlim(dataset_tick_positions[0] - 0.5, dataset_tick_positions[-1] + 0.5)
        ax.set_title(model_name.upper(), fontsize=12, fontweight='bold', pad=15)

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/pos_neg_combined/triple_model_independent_scales_ci.png", bbox_inches='tight')
    plt.show()

# use multiprocessing to speed up data grabbing
if __name__ == "__main__":
    # use multiprocessing for grab data function
    # model_names = ['paligemma']
    # image_dir = "2d_dataset_fixed_positions_1000"
    # save_dir = "data_saves_1000"

    # for model in model_names:
    #     # We start a fresh process for each model
    #     p = mp.Process(target=grab_2_data, args=(model, image_dir, save_dir))
    #     p.start()
        
    #     # p.join() is critical: it makes the script wait for the process to die 
    #     # (and free VRAM) before starting the next model.
    #     p.join()

    generate_bar_graphs_all_models_ci("data_saves_1000")

    #grab_2_data("paligemma", "whatsup", "data_saves_1000")