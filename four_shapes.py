from plotter2 import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, InternVLForConditionalGeneration    

def four_position(pos, neg=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global _loaded_models
    if '_loaded_models' not in globals():
        _loaded_models = {}

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


    item1_attn = []
    item2_attn = []
    item3_attn = []
    item4_attn = []
    baseline_attentions = []
    for i in range(5):
        image_path = f"4_shapes_same_dataset_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
        print("running image:", image_path)
        
        top_left_colour = plotter.get_shape_by_position('top_left').colour
        bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour 
        top_right_colour = plotter.get_shape_by_position('top_right').colour
        bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour

        if pos == 'top_left':
            target_colour = top_left_colour
        elif pos == 'bottom_left':
            target_colour = bottom_left_colour
        elif pos == 'top_right':
            target_colour = top_right_colour
        elif pos == 'bottom_right':
            target_colour = bottom_right_colour

        plotter.set_model(model, processor)
        
        if neg==False:
            plotter.get_outputs(f"The figure is {target_colour}")
        else:
            plotter.get_outputs(f"The figure is not {target_colour}")

        print("outputs:", plotter.print_output())
        bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
        item1_attn.append(bbox_attentions[top_left_colour])
        item2_attn.append(bbox_attentions[top_right_colour])
        item3_attn.append(bbox_attentions[bottom_left_colour])
        item4_attn.append(bbox_attentions[bottom_right_colour])
        baseline_attentions.append(baseline_attn)

    # plot average attention across items
    avg_item1_attn = np.mean(np.array(item1_attn), axis=0)
    avg_item2_attn = np.mean(np.array(item2_attn), axis=0)
    avg_item3_attn = np.mean(np.array(item3_attn), axis=0)
    avg_item4_attn = np.mean(np.array(item4_attn), axis=0)
    avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)
    layers = list(range(36))
    plt.figure(figsize=(10, 6))
    plt.plot(layers, avg_item1_attn, label='Top Left Shape Attention')
    plt.plot(layers, avg_item2_attn, label='Top Right Shape Attention')
    plt.plot(layers, avg_item3_attn, label='Bottom Left Shape Attention')
    plt.plot(layers, avg_item4_attn, label='Bottom Right Shape Attention')
    plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.ylim(0, 0.01)
    plt.title('Average Attention Scores Through Layers')
    plt.legend()
    # save figure with neg in filename if neg is True
    if neg:
        plt.savefig(f'intern_plots/4_shapes_{pos}_neg_all.png')
    else:
        plt.savefig(f'intern_plots/4_shapes_{pos}_all.png')

four_position('top_left')
four_position('top_right')
four_position('bottom_left')
four_position('bottom_right')
four_position('top_left', neg=True)
four_position('top_right', neg=True)
four_position('bottom_left', neg=True)
four_position('bottom_right', neg=True)