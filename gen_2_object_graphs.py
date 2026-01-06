from plotter2 import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration

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

    return processor, model

def four_position(model_name, pos, colour=None):
   
    processor, model = load_model(model_name)

    count = 0
    item1_attn = []
    item2_attn = []
    baseline_attentions = []
    for i in range(100):
        image_path = f"2d_dataset_fixed_positions_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour
        plotter.set_model(model, processor)

        if pos == 'left' or pos == 'neg_left':
            target_colour = left_colour
        elif pos == 'right' or pos == 'neg_right':
            target_colour = right_colour

        if target_colour == colour or colour is None:

            if pos == 'left':
                plotter.get_outputs(f"The figure is {left_colour}")
            elif pos == 'right':
                plotter.get_outputs(f"The figure is {right_colour}")
            elif pos == 'neg_left':
                plotter.get_outputs(f"The figure is not {left_colour}")
            elif pos == 'neg_right':
                plotter.get_outputs(f"The figure is not {right_colour}")

            print("outputs:", plotter.print_output())
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            item1_attn.append(bbox_attentions[left_colour])
            item2_attn.append(bbox_attentions[right_colour])
            baseline_attentions.append(baseline_attn)
            count += 1


    # plot average attention across items
    avg_item1_attn = np.mean(np.array(item1_attn), axis=0)
    avg_item2_attn = np.mean(np.array(item2_attn), axis=0)
    avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)
    std_item1_attn = np.std(np.array(item1_attn), axis=0)
    std_item2_attn = np.std(np.array(item2_attn), axis=0)
    layers = list(range(36))
    plt.figure(figsize=(10, 6))
    plt.plot(layers, avg_item1_attn, label='Left Shape Attention')
    plt.plot(layers, avg_item2_attn, label='Right Shape Attention')
    plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    # plot confidence intervals
    #plt.fill_between(layers, avg_item1_attn - std_item1_attn, avg_item1_attn + std_item1_attn, color='blue', alpha=0.2)
    #plt.fill_between(layers, avg_item2_attn - std_item2_attn, avg_item2_attn + std_item2_attn, color='red', alpha=0.2)    
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Average Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, 0.03)
    plt.savefig(f'intern_plots/{pos}_{colour+"_" if colour else ""}all.png')
    print(f"Processed {count} images matching criteria.")


if __name__ == "__main__":
    # four_position('internvl', 'left')
    # four_position('internvl', 'right')
    # four_position('internvl', 'neg_left')
    # four_position('internvl', 'neg_right')
    four_position('internvl', 'left', 'red')
    four_position('internvl', 'right', 'red')
    four_position('internvl', 'neg_left', 'red')
    four_position('internvl', 'neg_right', 'red')
    four_position('internvl', 'left', 'blue')
    four_position('internvl', 'right', 'blue')
    four_position('internvl', 'neg_left', 'blue')
    four_position('internvl', 'neg_right', 'blue')
    four_position('internvl', 'left', 'green')
    four_position('internvl', 'right', 'green')
    four_position('internvl', 'neg_left', 'green')
    four_position('internvl', 'neg_right', 'green')
    four_position('internvl', 'left', 'yellow')
    four_position('internvl', 'right', 'yellow')
    four_position('internvl', 'neg_left', 'yellow')
    four_position('internvl', 'neg_right', 'yellow')
