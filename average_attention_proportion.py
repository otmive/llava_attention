from plotter import Plotter
import torch 
import os 
from transformers import AutoProcessor, AutoModelForImageTextToText, InternVLForConditionalGeneration, LlavaForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import gc
from matplotlib import rcParams

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




def plot_all_layers(model_name, data_dir):
    mentioned_attn = []
    not_mentioned_attn = []
    mentioned_attn_neg = []
    not_mentioned_attn_neg = []
    baseline_attn = []
    shapes_attn = []
    processor, model = load_model(model_name)

    for pos in ['left', 'right', 'neg_left', 'neg_right']:
        item1_attn = []
        item2_attn = []
        for img_path in sorted(os.listdir(data_dir))[0:5]: # Adjust range as needed
            print("image path: ", img_path)
            plotter = Plotter(data_dir + "/" + img_path)
            plotter.set_model(model, processor)
            left_colour = plotter.get_left_shapes()[0].colour
            right_colour = plotter.get_right_shapes()[0].colour

            if pos == 'left':
                plotter.get_outputs(f"The figure is {left_colour}")
                bbox_attentions, baseline_attention = plotter.plot_attention_through_layers(save_path=f"bar_plots/test2_{pos}_{img_path}_{model_name}.png")
                total_attn = sum(bbox_attentions[left_colour]) + sum(bbox_attentions[right_colour]) + sum(baseline_attention)
                mentioned_attn.append(np.sum(bbox_attentions[left_colour]) / total_attn)
                not_mentioned_attn.append(np.sum(bbox_attentions[right_colour]) / total_attn)
                baseline_attn.append(np.sum(baseline_attention) / total_attn)
                shapes_attn.append((np.sum(bbox_attentions[left_colour]) + np.sum(bbox_attentions[right_colour])) / total_attn)
            elif pos == 'right':
                plotter.get_outputs(f"The figure {right_colour}")
                bbox_attentions, baseline_attention = plotter.plot_attention_through_layers(save_path=f"bar_plots/test_{pos}2_{img_path}_{model_name}.png")
                total_attn = sum(bbox_attentions[right_colour]) + sum(bbox_attentions[left_colour]) + sum(baseline_attention)
                mentioned_attn.append(np.sum(bbox_attentions[right_colour]) / total_attn)
                not_mentioned_attn.append(np.sum(bbox_attentions[left_colour]) / total_attn)   
                baseline_attn.append(np.sum(baseline_attention) / total_attn) 
                shapes_attn.append((np.sum(bbox_attentions[left_colour]) + np.sum(bbox_attentions[right_colour])) / total_attn)
            elif pos == 'neg_left':
                plotter.get_outputs(f"The figure is not {left_colour}")
                bbox_attentions, baseline_attention = plotter.plot_attention_through_layers(save_path=f"bar_plots/test2_{pos}_{img_path}_{model_name}.png")
                total_attn = sum(bbox_attentions[left_colour]) + sum(bbox_attentions[right_colour]) + sum(baseline_attention)
                mentioned_attn_neg.append(np.sum(bbox_attentions[left_colour]) / total_attn)
                not_mentioned_attn_neg.append(np.sum(bbox_attentions[right_colour]) / total_attn)
                baseline_attn.append(np.sum(baseline_attention) / total_attn)
                shapes_attn.append((np.sum(bbox_attentions[left_colour]) + np.sum(bbox_attentions[right_colour])) / total_attn)
            elif pos == 'neg_right':
                plotter.get_outputs(f"The figure is not {right_colour}")
                bbox_attentions, baseline_attention = plotter.plot_attention_through_layers(save_path=f"bar_plots/test2_{pos}_{img_path}_{model_name}.png")
                total_attn = sum(bbox_attentions[right_colour]) + sum(bbox_attentions[left_colour]) + sum(baseline_attention)
                mentioned_attn_neg.append(np.sum(bbox_attentions[right_colour]) / total_attn)
                not_mentioned_attn_neg.append(np.sum(bbox_attentions[left_colour]) / total_attn)
                baseline_attn.append(np.sum(baseline_attention) / total_attn)
                shapes_attn.append((np.sum(bbox_attentions[left_colour]) + np.sum(bbox_attentions[right_colour])) / total_attn)


    # calculate 95% confidence interval for each category
    sem_mentioned = np.std(mentioned_attn) / np.sqrt(len(mentioned_attn))
    sem_not_mentioned = np.std(not_mentioned_attn) / np.sqrt(len(not_mentioned_attn))
    sem_mentioned_neg = np.std(mentioned_attn_neg) / np.sqrt(len(mentioned_attn_neg))
    sem_not_mentioned_neg = np.std(not_mentioned_attn_neg) / np.sqrt(len(not_mentioned_attn_neg ))
    ci_mentioned = 1.96 * sem_mentioned
    ci_not_mentioned = 1.96 * sem_not_mentioned
    ci_mentioned_neg = 1.96 * sem_mentioned_neg
    ci_not_mentioned_neg = 1.96 * sem_not_mentioned_neg

    print("Average mentioned attention (positive): ", np.mean(mentioned_attn))
    print("Average not mentioned attention (positive): ", np.mean(not_mentioned_attn))
    print("Average mentioned attention (negative): ", np.mean(mentioned_attn_neg))
    print("Average not mentioned attention (negative): ", np.mean(not_mentioned_attn_neg))
    print("Average baseline attention: ", np.mean(baseline_attn))
    print("Average shapes attention: ", np.mean(shapes_attn))
    final_data = {
        'pos_means': [np.mean(mentioned_attn), np.mean(not_mentioned_attn)], # mentioned, not mentioned
        'pos_cis': [ci_mentioned, ci_not_mentioned],
        'neg_means': [np.mean(mentioned_attn_neg), np.mean(not_mentioned_attn_neg)], # mentioned, not mentioned
        'neg_cis': [ci_mentioned_neg, ci_not_mentioned_neg]
    }


    return final_data

import numpy as np
import matplotlib.pyplot as plt
import gc
import torch
import os



def plot_all_layers_4(model_name, data_dir):
    # We only need to iterate through the target locations
    base_positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    # Store results: { 'pos': [[mentioned_vals], [not_mentioned_vals]], 'neg': [[...], [...]] }
    # Index 0 = Mentioned, Index 1 = Not Mentioned
    results = {
        'positive': [[] for _ in range(2)],
        'negative': [[] for _ in range(2)]
    }
    
    processor, model_obj = load_model(model_name)
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 1. Process data
    for pos in base_positions:
        # Check both the standard prompt and the negative prompt for each position
        for is_negative in [False, True]:
            state_key = 'negative' if is_negative else 'positive'
            
            for img_path in image_files[0:5]: # Adjust range as needed
                try:
                    plotter = Plotter(os.path.join(data_dir, img_path))
                    plotter.set_model(model_obj, processor)
                    
                    # Map positions to their actual colors in this specific image
                    shapes = {p: plotter.get_shape_by_position(p).colour for p in base_positions}
                    
                    # The color being talked about (the "Mentioned" color)
                    mentioned_color = shapes[pos]
                    prompt = f"The figure is {'not ' if is_negative else ''}{mentioned_color}"
                    
                    plotter.get_outputs(prompt)
                    bbox_attentions, baseline_attn = plotter.plot_attention_through_layers_total("tmp_plot")
                    total_attn = sum(bbox_attentions['red']) + sum(bbox_attentions['green']) + sum(bbox_attentions['blue']) + sum(bbox_attentions['yellow']) + sum(baseline_attn)


                    # 2. Distinguish Mentioned vs Not Mentioned
                    for p_key in base_positions:
                        color = shapes[p_key]
                        if color in bbox_attentions:
                            attn_list = np.array(bbox_attentions[color])
                            avg_attn = np.sum(attn_list) / total_attn
                            
                            if p_key == pos:
                                # This is the object the prompt described
                                results[state_key][0].append(avg_attn)
                            else:
                                # This is one of the other 3 objects
                                results[state_key][1].append(avg_attn)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue

    # 3. Calculate Means
    sem_mentioned_pos = np.std(results['positive'][0]) / np.sqrt(len(results['positive'][0])) if results['positive'][0] else 0
    sem_not_mentioned_pos = np.std(results['positive'][1]) / np.sqrt(len(results['positive'][1])) if results['positive'][1] else 0
    sem_mentioned_neg = np.std(results['negative'][0]) / np.sqrt(len(results['negative'][0])) if results['negative'][0] else 0
    sem_not_mentioned_neg = np.std(results['negative'][1]) / np.sqrt(len(results['negative'][1])) if results['negative'][1] else 0

    ci_mentioned_pos = 1.96 * sem_mentioned_pos
    ci_not_mentioned_pos = 1.96 * sem_not_mentioned_pos
    ci_mentioned_neg = 1.96 * sem_mentioned_neg
    ci_not_mentioned_neg = 1.96 * sem_not_mentioned_neg
    final_data = {
        'pos_means': [np.mean(results['positive'][0]) if results['positive'][0] else 0, # mentioned
                      np.mean(results['positive'][1]) if results['positive'][1] else 0], # not mentioned
        'pos_cis': [ci_mentioned_pos, ci_not_mentioned_pos],
        'neg_means': [np.mean(results['negative'][0]) if results['negative'][0] else 0, # mentioned
                      np.mean(results['negative'][1]) if results['negative'][1] else 0], # not mentioned
        'neg_cis': [ci_mentioned_neg, ci_not_mentioned_neg]
    }   

    return final_data


import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

# --- 1. The Worker Function (Runs on GPU) ---
def model_worker(model_name, img_folder_2, img_folder_4, queue):
    """
    Loads the model ONCE, runs both 2-shape and 4-shape analysis,
    then returns both sets of data.
    """
    print(f"--- Loading {model_name} ---")
    # model = load_your_model(model_name) 

    # 1. Run 2-shape analysis
    # Assuming plot_all_layers returns the 'final_data' dict you showed me
    data_2 = plot_all_layers(model_name, img_folder_2) 
    
    # 2. Run 4-shape analysis
    data_4 = plot_all_layers_4(model_name, img_folder_4)
    
    # Send both back as a tuple/dict
    queue.put({
        '2_shapes': data_2,
        '4_shapes': data_4
    })
    print(f"--- {model_name} Complete ---")

# --- 2. The Main Process ---
if __name__ == "__main__":
    models = ['llava', 'internvl', 'paligemma']
    folder_2 = "2d_dataset_fixed_positions_1000/images"
    folder_4 = "4_shapes_same_dataset_1000/images"
    
    # This will store everything: { 'llava': {'2_shapes': {...}, '4_shapes': {...}}, ... }
    master_data = {}

    for m_name in models:
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=model_worker, args=(m_name, folder_2, folder_4, q))
        
        p.start()
        master_data[m_name] = q.get() # Get the combined dict
        p.join() # GPU is now cleared for the next model

    # --- 3. Plotting Section ---
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7))
    
    # # We define the labels and colors once
    # categories = ['Pos Mentioned', 'Pos Not Mentioned', 'Neg Mentioned', 'Neg Not Mentioned']
    # colors = ['#2D5A27', "#6DB459", '#8B0000', '#FF6B6B'] # Deep Blue, Teal, Deep Red, Bright Red

    # def draw_bars(ax, results_key, title):
    #     x = np.arange(len(models))
    #     width = 0.18
        
    #     # Pull the specific data (either '2_shapes' or '4_shapes') for each model
    #     pos_m  = [master_data[m][results_key]['pos_means'][0] for m in models] # mentioned
    #     pos_nm = [master_data[m][results_key]['pos_means'][1] for m in models] # not mentioned
    #     neg_m  = [master_data[m][results_key]['neg_means'][0] for m in models] # mentioned
    #     neg_nm = [master_data[m][results_key]['neg_means'][1] for m in models] # not mentioned

    #     ax.bar(x - 1.5*width, pos_m,  width, label=categories[0], color=colors[0])
    #     ax.bar(x - 0.5*width, pos_nm, width, label=categories[1], color=colors[1])
    #     ax.bar(x + 0.5*width, neg_m,  width, label=categories[2], color=colors[2])
    #     ax.bar(x + 1.5*width, neg_nm, width, label=categories[3], color=colors[3])

    #     ax.set_title(title, fontsize=18, fontweight='bold')
    #     ax.set_xticks(x)
    #     ax.set_xticklabels([m.upper() for m in models])
    #     ax.legend(fontsize=14)
    #     ax.set_ylabel('Average Attention', fontsize=14)
    #     ax.set_xlabel('Model', fontsize=14)

    # # Create the two rectangular plots
    # draw_bars(ax1, '2_shapes', "Attention: 2 Shapes")
    # draw_bars(ax2, '4_shapes', "Attention: 4 Shapes")

    # plt.tight_layout()
    # plt.savefig("plots/2d_datasets_2/bar_plots.png")


    # Set publication-quality defaults
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42

    # Professional color palette with better contrast and accessibility
    colors = {
        'pos_mentioned': '#2E7D32',      # Deep green
        'pos_not_mentioned': '#81C784',  # Light green
        'neg_mentioned': '#C62828',      # Deep red
        'neg_not_mentioned': "#F36967"   # Light red
    }

    def draw_point_plots(ax, results_key, title):
        """
        Draw grouped point plots with error bars for model comparison.
        
        Parameters:
        -----------
        ax : matplotlib axis
            The axis to draw on
        results_key : str
            Key to access the results in master_data
        title : str
            Plot title
        """
        x = np.arange(len(models))
        width = 0.20
        
        labels = ['Pos Mentioned', 'Pos Not Mentioned', 'Neg Mentioned', 'Neg Not Mentioned']
        color_keys = ['pos_mentioned', 'pos_not_mentioned', 'neg_mentioned', 'neg_not_mentioned']
        offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
        
        # Plot each model's data points
        for i, m_name in enumerate(models):
            data = master_data[m_name][results_key]
            means = data['pos_means'] + data['neg_means']
            cis = data['pos_cis'] + data['neg_cis']
            
            for j in range(4):
                ax.errorbar(
                    x[i] + offsets[j], 
                    means[j], 
                    yerr=cis[j],
                    fmt='o',
                    color=colors[color_keys[j]],
                    label=labels[j] if i == 0 else "",
                    markersize=12,
                    capsize=6,
                    capthick=2.5,
                    elinewidth=2.5,
                    markeredgewidth=1.5,
                    markeredgecolor='white',
                    zorder=10,
                    alpha=0.9
                )
        
        # Title and labels with clear hierarchy
        ax.set_title(title, fontsize=22, fontweight='bold', pad=25, color='black')
        ax.set_xlabel('Model', fontsize=18, fontweight='600', labelpad=12, color='black')
        ax.set_ylabel('Proportion of Total Attention', fontsize=18, fontweight='600', labelpad=12, color='black')
        
        # X-axis configuration
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models], fontsize=16, fontweight='500')
        
        # Y-axis configuration with better readability
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=5, colors='black')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.4f}'))
        
        # Clean, minimal styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')
        
        # Subtle grid for easier reading
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=1.2, color='black', zorder=0)
        ax.set_axisbelow(True)
        
        # Add legend only to the second plot
        if "4 Shapes" in title:
            legend = ax.legend(
                bbox_to_anchor=(1.02, 1), 
                loc='upper left',
                fontsize=15,
                handletextpad=0.8,
                borderpad=1.2,
                frameon=False,
                fancybox=False,
                shadow=False
            )



    # Main plotting function
    def create_professional_plot(master_data, models, output_path="plots/2d_datasets_2/point_plots_proportion_ci_test.png"):
        """
        Create a publication-ready comparison plot.
        
        Parameters:
        -----------
        master_data : dict
            Dictionary containing the experimental results
        models : list
            List of model names
        output_path : str
            Where to save the figure
        """
        # Calculate shared Y-limit based on actual data
        max_val = 0
        for m_name in models:
            for key in ['2_shapes', '4_shapes']:
                data = master_data[m_name][key]
                means = data['pos_means'] + data['neg_means']
                cis = data['pos_cis'] + data['neg_cis']
                max_val = max(max_val, max([m + c for m, c in zip(means, cis)]))
        
        # Add 10% padding to max value
        max_val *= 1.1
        
        # Create figure with appropriate size
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.patch.set_facecolor('white')
        
        # Draw both subplots
        draw_point_plots(ax1, '2_shapes', "2 Shapes")
        draw_point_plots(ax2, '4_shapes', "4 Shapes")
        
        # Synchronize y-axes for fair comparison
        ax1.set_ylim(0, max_val)
        ax2.set_ylim(0, max_val)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout(pad=2.5)
        
        # Save with high quality
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Plot saved to: {output_path}")
        
        
    create_professional_plot(master_data, models)

