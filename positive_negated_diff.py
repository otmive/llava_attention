from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def plot_diff(target_colour):

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
    baseline_attentions = []
    with torch.inference_mode():
        for i in range(1000):
            image_path = f"2d_dataset_fixed_positions_1000/images/image_{i:04d}.png"
            plotter = Plotter(image_path)
            left_colour = plotter.get_left_shapes()[0].colour
            right_colour = plotter.get_right_shapes()[0].colour
            if right_colour == target_colour:
                plotter.set_model(model, processor)
                plotter.get_outputs(f"The figure is {right_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                item1_attn_pos = bbox_attentions[left_colour]
                item2_attn_pos = bbox_attentions[right_colour]
                # baseline_attentions.append(baseline_attn)
                plotter.get_outputs(f"The figure is not {right_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                item1_attn_neg = bbox_attentions[left_colour]
                item2_attn_neg = bbox_attentions[right_colour]
                # baseline_attentions.append(baseline_attn)
                # calculate differnece betwen item1_attn_pos and item1_attn_neg
                item1_diff_attn.append(np.array(item1_attn_neg) - np.array(item1_attn_pos))
                item2_diff_attn.append(np.array(item2_attn_neg) - np.array(item2_attn_pos))


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
    x = np.array(list(range(32)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, avg_item1_attn, width, label='Left Shape (mentioned)')
    ax.bar(x + width/2, avg_item2_attn, width, label='Right Shape (not mentioned)')
    # set x axis
    ax.set_ylim(-0.0014, 0.0014)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Attention Difference')
    ax.set_title('Average Attention Difference (Negated - Positive) Through Layers')
    ax.legend()
    plt.savefig(f'positive_negative_diff_graphs/2_shape_{target_colour}_right_positive_negated_diff.png')
if __name__ == "__main__":
    plot_diff('red')
    plot_diff('blue')
    plot_diff('green')
    plot_diff('yellow')



