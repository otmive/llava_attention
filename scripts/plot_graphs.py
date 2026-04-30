import matplotlib.pyplot as plt
import numpy as np
from functions import load_model, mean_and_ci
import torch
import gc
import multiprocessing as mp
from plotter import Plotter
import argparse



def generate_bar_graphs_all_models_ci(save_dir):

    model_names = ['paligemma', 'internvl', 'llava']
    dataset_types = ["binary", "multary", "whatsup"]
    dataset_labels = ["Bin.", "Mul.", "What's Up"]
    
    color_mentioned = '#2c7fb8'     
    color_non_mentioned = '#f03b20' 
    bar_width = 0.18            
    group_spacing = 0.2
    CI_Z = 1.96  # 95% confidence interval


    fig, axes = plt.subplots(1, 3, figsize=(11, 3), dpi=120, sharey=False)
    fig.subplots_adjust(wspace=0.3)

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        dataset_tick_positions = []

        for j, dataset_type in enumerate(dataset_types):
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

            means_cis = [mean_and_ci(arr.flatten()) for arr in [m_pos, m_neg, nm_pos, nm_neg]]
            vals  = [mc[0] for mc in means_cis]
            yerrs = [mc[1] for mc in means_cis]
            
            base_x = j * (4 * bar_width + group_spacing)
            center = base_x + (1.5 * bar_width)
            dataset_tick_positions.append(center)

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
                    zorder=5             
                )

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
    plt.savefig(f"plots/triple_model_independent_scales_ci.png", bbox_inches='tight')

def plot_left_right(model_name, image_path):
    print("loading model ", model_name)
    processor, model = load_model(model_name)
    if "whatsup" in image_path:
      noun = "object"
    else:
      noun = "shape"
    plotter = Plotter(image_path)
    plotter.set_model(model, processor)
    left_colour = plotter.get_left_shapes()[0].colour
    right_colour = plotter.get_right_shapes()[0].colour
    plotter.get_outputs(f"The {noun} is {left_colour}")
    l_bbox_attentions, l_baseline_attn = plotter.plot_attention_through_layers()
    plotter.get_outputs(f"The {noun} is {right_colour}")
    r_bbox_attentions, r_baseline_attn = plotter.plot_attention_through_layers()
    plotter.get_outputs(f"The {noun} is not {left_colour}")
    ln_bbox_attentions, ln_baseline_attn = plotter.plot_attention_through_layers()
    plotter.get_outputs(f"The {noun} is not {right_colour}")
    rn_bbox_attentions, rn_baseline_attn = plotter.plot_attention_through_layers()

    # get max value of both left and right attentions
    max_left_attn = max([max(attn) for attn in l_bbox_attentions.values()])
    max_right_attn = max([max(attn) for attn in r_bbox_attentions.values()])
    max_left_neg_attn = max([max(attn) for attn in ln_bbox_attentions.values()])
    max_right_neg_attn = max([max(attn) for attn in rn_bbox_attentions.values()])
    overall_max = max(max_left_attn, max_right_attn, max_left_neg_attn, max_right_neg_attn  )

    num_layers = len(l_baseline_attn)
    layers = list(range(num_layers))
    # plot the attentions for the left colour in a plot
    plt.figure(figsize=(10, 6))
    if left_colour in ['red', 'blue', 'green', 'yellow']:
        left_plot_colour = left_colour if left_colour in ['red', 'blue', 'green'] else 'gold'
    else:
        left_plot_colour = None
    if right_colour in ['red', 'blue', 'green', 'yellow']:
        right_plot_colour = right_colour if right_colour in ['red', 'blue', 'green'] else 'gold'
    else:
        right_plot_colour = None
    plt.plot(layers, l_bbox_attentions[left_colour], label=f'Left Shape Attention ({left_colour})', color=left_plot_colour)
    plt.plot(layers, l_bbox_attentions[right_colour], label=f'Right Shape Attention ({right_colour})', color=right_plot_colour)
    plt.plot(layers, l_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, overall_max * 1.1)  
    plt.savefig(f'plots/{model_name}_left_{image_path.split("/")[-1].split(".")[0]}_attention.png')

    # plot the attentions for the right colour in a plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, r_bbox_attentions[left_colour], label=f'Left Shape Attention ({left_colour})', color=left_plot_colour)
    plt.plot(layers, r_bbox_attentions[right_colour], label=f'Right Shape Attention ({right_colour})', color=right_plot_colour)
    plt.plot(layers, r_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, overall_max * 1.1)  
    plt.savefig(f'plots/{model_name}_right_{image_path.split("/")[-1].split(".")[0]}_attention.png')
    
    # plot negated versions
    plt.figure(figsize=(10, 6))
    plt.plot(layers, ln_bbox_attentions[left_colour], label=f'Left Shape Attention ({left_colour})', color=left_plot_colour)
    plt.plot(layers, ln_bbox_attentions[right_colour], label=f'Right Shape Attention ({right_colour})', color=right_plot_colour)
    plt.plot(layers, ln_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, overall_max * 1.1)  
    plt.savefig(f'plots/{model_name}_neg_left_{image_path.split("/")[-1].split(".")[0]}_attention.png')

    # plot the attentions for the right colour in a plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, rn_bbox_attentions[left_colour], label=f'Left Shape Attention ({left_colour})', color=left_plot_colour)
    plt.plot(layers, rn_bbox_attentions[right_colour], label=f'Right Shape Attention ({right_colour})', color=right_plot_colour)
    plt.plot(layers, rn_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, overall_max * 1.1)  
    plt.savefig(f'plots/{model_name}_neg_right_{image_path.split("/")[-1].split(".")[0]}_attention.png')
    # free up memory
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Finished plotting for ", model_name)

def generate_layerwise(data_type, save_dir):
        fig, axs = plt.subplots(2, 3, figsize=(21, 8))
        if data_type == "binary":
            data_name = "Binary"
        elif data_type == "multary":
            data_name = "Multary"
        else:
            data_name = "What's Up"
        # add title to entire figure
        fig.suptitle(f"{data_name}", fontsize=20, fontweight='bold')#, y=1.02
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
                mentioned_attn = np.load(f"{save_dir}/{model_name}_{data_type}_{neg_val}mentioned_attn.npy")
                not_mentioned_attn = np.load(f"{save_dir}/{model_name}_{data_type}_{neg_val}non_mentioned_attn.npy")
                if data_type == "multary":
                    avg_m = np.mean(mentioned_attn, axis=0)
                    print(avg_m.shape)
                    avg_nm = np.mean(not_mentioned_attn, axis=(0,1))
                    ci_m = np.std(mentioned_attn, axis=(0)) / np.sqrt(len(mentioned_attn)) * 1.96
                    ci_nm = np.std(not_mentioned_attn, axis=(0,1)) / np.sqrt(len(not_mentioned_attn)) * 1.96
                else:   
                
                    avg_m = np.mean(mentioned_attn, axis=0)
                    print(f"{save_dir}/{model_name}_{data_type}_{neg_val}mentioned_attn.npy")
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
        plt.savefig(f"plots/{data_type}.png", dpi=300)

def position_graphs(model_name, save_dir):
    positions = ["left", "neg_left", "right", "neg_right"]
    
    title_map = {
        "left": "Affirmative Left",
        "right": "Affirmative Right",
        "neg_left": "Negated Left",
        "neg_right": "Negated Right"
    }
    # 3. Create the figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle(f"{model_name.upper()}", fontsize=16, fontweight='bold')
    
    all_values = []
    # load data from data_saves/left_right_data/
    for i, pos in enumerate(positions):
        item1_attn = np.load(f"{save_dir}/{model_name}_binary_left_item_attn_{pos}.npy")
        item2_attn = np.load(f"{save_dir}/{model_name}_binary_right_item_attn_{pos}.npy")

        avg_item1_attn = np.mean(item1_attn, axis=0)
        avg_item2_attn = np.mean(item2_attn, axis=0)
        ci1 = 1.96*(np.std(item1_attn, axis=0, ddof=1) / np.sqrt(len(item1_attn)))
        ci2 = 1.96*(np.std(item2_attn, axis=0, ddof=1) / np.sqrt(len(item2_attn)))

        all_values.extend(avg_item1_attn + ci1)
        all_values.extend(avg_item2_attn + ci2)

        x = list(range(len(avg_item1_attn)))
        axes[i].plot(avg_item1_attn, label='Left Shape', linewidth=2, marker='o', markersize=3)
        axes[i].plot(avg_item2_attn, label='Right Shape', linewidth=2, marker='x', markersize=3)    
        axes[i].fill_between(x, avg_item1_attn - ci1, avg_item1_attn + ci1, alpha=0.25)
        axes[i].fill_between(x, avg_item2_attn - ci2, avg_item2_attn + ci2, alpha=0.25) 
        axes[i].set_title(f"{title_map.get(pos, pos)}", fontsize=14)
        axes[i].set_xlabel("Layer", fontsize=16)
        # make tick label font size bigger
        axes[i].tick_params(axis='both', which='major', labelsize=16)
        axes[i].set_ylim(0, max(all_values)*1.1)  # Apply the calculated max limit  
        if i == 0:
            axes[i].set_ylabel("Attention", fontsize=16)
        if i == 3:
            axes[i].legend(fontsize=16)
        axes[i].grid(True, linestyle='--', alpha=0.4)   
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f"plots/{model_name}_position_graphs.png"
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Plot saved as {output_filename}")

if __name__ == "__main__":

    #plot_left_right("llava", "binary/images/image_0000.png")

    generate_bar_graphs_all_models_ci("data_saves")

    data_types = ["binary", "multary", "whatsup"]
    for data_type in data_types:
      generate_layerwise(data_type, "data_saves")

    models = ["llava", "internvl", "paligemma"]
    for model in models:
      position_graphs(model, "data_saves")




