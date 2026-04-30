import matplotlib.pyplot as plt
import numpy as np
from functions import load_model
import torch
import gc
import multiprocessing as mp
from plotter import Plotter




def generate_bar_graphs_all_models_ci(save_dir):

    model_names = ['paligemma', 'internvl', 'llava']
    dataset_types = ["2_obj", "4_obj", "whatsup"]
    dataset_labels = ["Bin.", "Mul.", "What's Up"]
    
    color_mentioned = '#2c7fb8'     
    color_non_mentioned = '#f03b20' 
    bar_width = 0.18            
    group_spacing = 0.2
    CI_Z = 1.96  # 95% confidence interval

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

if __name__ == "__main__":
  plot_left_right("llava", "binary/images/image_0000.png")