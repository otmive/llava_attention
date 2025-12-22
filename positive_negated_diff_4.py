from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def plot_diff(pos):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global _loaded_models
    if '_loaded_models' not in globals():
        _loaded_models = {}

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


    item1_diff_attn = []
    item2_diff_attn = []
    item3_diff_attn = []
    item4_diff_attn = []
    baseline_attentions = []
    for i in range(100):
        image_path = f"4_shapes_same_dataset_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
        top_left_colour = plotter.get_shape_by_position('top_left').colour
        top_right_colour = plotter.get_shape_by_position('top_right').colour
        bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
        bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour
        if pos == 'top_left':
            target_colour = top_left_colour
        elif pos == 'top_right':
            target_colour = top_right_colour
        elif pos == 'bottom_left':
            target_colour = bottom_left_colour
        elif pos == 'bottom_right':
            target_colour = bottom_right_colour


        plotter.set_model(model, processor)
        plotter.get_outputs(f"The figure is {target_colour}")
        bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
        item1_attn_pos = bbox_attentions[top_left_colour]
        item2_attn_pos = bbox_attentions[top_right_colour]
        item3_attn_pos = bbox_attentions[bottom_left_colour]
        item4_attn_pos = bbox_attentions[bottom_right_colour]
        # baseline_attentions.append(baseline_attn)
        plotter.get_outputs(f"The figure is not {target_colour}")
        bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
        item1_attn_neg = bbox_attentions[top_left_colour]
        item2_attn_neg = bbox_attentions[top_right_colour]
        item3_attn_neg = bbox_attentions[bottom_left_colour]
        item4_attn_neg = bbox_attentions[bottom_right_colour]
        # baseline_attentions.append(baseline_attn)
        # calculate differnece betwen item1_attn_pos and item1_attn_neg
        item1_diff_attn.append(np.array(item1_attn_neg) - np.array(item1_attn_pos))
        item2_diff_attn.append(np.array(item2_attn_neg) - np.array(item2_attn_pos))
        item3_diff_attn.append(np.array(item3_attn_neg) - np.array(item3_attn_pos))
        item4_diff_attn.append(np.array(item4_attn_neg) - np.array(item4_attn_pos))

    # plot difference between positive and negated attention 
    avg_item1_attn = np.mean(np.array(item1_diff_attn), axis=0)
    avg_item2_attn = np.mean(np.array(item2_diff_attn), axis=0)
    avg_item3_attn = np.mean(np.array(item3_diff_attn), axis=0)
    avg_item4_attn = np.mean(np.array(item4_diff_attn), axis=0)
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
    x = np.array(list(range(32)))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, avg_item1_attn, width, label=f"Top Left Shape {'(mentioned)' if pos=='top_left' else '(not mentioned)'}")
    ax.bar(x - 0.5*width, avg_item2_attn, width, label=f"Top Right Shape {'(mentioned)' if pos=='top_right' else '(not mentioned)'}")
    ax.bar(x + 0.5*width, avg_item3_attn, width, label=f"Bottom Left Shape {'(mentioned)' if pos=='bottom_left' else '(not mentioned)'}")
    ax.bar(x + 1.5*width, avg_item4_attn, width, label=f"Bottom Right Shape {'(mentioned)' if pos=='bottom_right' else '(not mentioned)'}")
    for i in range(32):
        ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_xlim(-0.5, 31.5) 
    ax.set_ylim(-0.0012, 0.0012)
    ax.set_ylabel('Attention Difference')
    ax.set_title('Average Attention Difference (Negated - Positive) Through Layers')
    ax.legend()
    plt.savefig(f'positive_negative_diff_graphs/4_shape_{pos}_positive_negated_diff.png')
if __name__ == "__main__":
    plot_diff('top_left')
    plot_diff('top_right')
    plot_diff('bottom_left')
    plot_diff('bottom_right')




