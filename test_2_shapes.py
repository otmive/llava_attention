from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def plot_attention_colour(colour):

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


    item1_attn = []
    item2_attn = []
    baseline_attentions = []
    for i in range(1000):
        image_path = f"2d_dataset_fixed_positions_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour
        if right_colour == colour:
            plotter.set_model(model, processor)
            plotter.get_outputs(f"The figure is not {right_colour}")
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            item1_attn.append(bbox_attentions[left_colour])
            item2_attn.append(bbox_attentions[right_colour])
            baseline_attentions.append(baseline_attn)

    # plot average attention across items
    avg_item1_attn = np.mean(np.array(item1_attn), axis=0)
    avg_item2_attn = np.mean(np.array(item2_attn), axis=0)
    avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)
    layers = list(range(32))
    plt.figure(figsize=(10, 6))
    plt.plot(layers, avg_item1_attn, label='Left Shape Attention')
    plt.plot(layers, avg_item2_attn, label='Right Shape Attention')
    plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Average Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, 0.007)
    plt.savefig(f'Images/right_{colour}_negated_2_shapes_average_attention.png')

if __name__ == "__main__":
    plot_attention_colour('green')
    plot_attention_colour('red')
    plot_attention_colour('blue')
    plot_attention_colour('yellow')


