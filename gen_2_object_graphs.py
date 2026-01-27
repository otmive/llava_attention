from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration
from transformers import AutoModelForImageTextToText

HF_TOKEN = "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"

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
            processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.float16, token=HF_TOKEN).to(device)
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
        image_path = f"2d_dataset_right_large_1000/images/image_{i:04d}.png"
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
    if model_name == 'llava':
        num_layers = 32
    elif model_name == 'internvl':
        num_layers = 36
    elif model_name == 'paligemma':
        num_layers = 26
    layers = list(range(num_layers))

    plt.figure(figsize=(10, 6))
    plt.plot(layers, avg_item1_attn, label='Left Shape Attention')
    plt.plot(layers, avg_item2_attn, label='Right Shape Attention')
    plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    # plot confidence intervals
    plt.fill_between(layers, avg_item1_attn - std_item1_attn, avg_item1_attn + std_item1_attn, color='blue', alpha=0.2)
    plt.fill_between(layers, avg_item2_attn - std_item2_attn, avg_item2_attn + std_item2_attn, color='red', alpha=0.2)    
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Average Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    #plt.ylim(0, 0.018)
    plt.ylim(0, 0.005)
    #plt.ylim(0, 0.04)
    plt.savefig(f'right_large_plots/{pos}_{colour+"_" if colour else ""}{model_name}.png')
    print(f"Processed {count} images matching criteria.")


if __name__ == "__main__":
    four_position('llava', 'left')
    four_position('llava', 'right')
    four_position('llava', 'neg_left')
    four_position('llava', 'neg_right')
    four_position('llava', 'left')
    four_position('llava', 'right')
    four_position('llava', 'neg_left')
    four_position('llava', 'neg_right')
    # four_position('llava', 'left', 'red')
    # four_position('llava', 'right', 'red')
    # four_position('llava', 'neg_left', 'red')
    # four_position('llava', 'neg_right', 'red')
    # four_position('llava', 'left', 'blue')
    # four_position('llava', 'right', 'blue')
    # four_position('llava', 'neg_left', 'blue')
    # four_position('llava', 'neg_right', 'blue')
    # four_position('llava', 'left', 'green')
    # four_position('llava', 'right', 'green')
    # four_position('llava', 'neg_left', 'green')
    # four_position('llava', 'neg_right', 'green')
    # four_position('llava', 'left', 'yellow')
    # four_position('llava', 'right', 'yellow')
    # four_position('llava', 'neg_left', 'yellow')
    # four_position('llava', 'neg_right', 'yellow')
    # four_position('paligemma', 'left')
    # four_position('paligemma', 'right')
    # four_position('paligemma', 'neg_left')
    # four_position('paligemma', 'neg_right')
    # four_position('paligemma', 'left', 'red')
    # four_position('paligemma', 'right', 'red')
    # four_position('paligemma', 'neg_left', 'red')
    # four_position('paligemma', 'neg_right', 'red')
    # four_position('paligemma', 'left', 'blue')
    # four_position('paligemma', 'right', 'blue')
    # four_position('paligemma', 'neg_left', 'blue')
    # four_position('paligemma', 'neg_right', 'blue')
    # four_position('paligemma', 'left', 'green')
    # four_position('paligemma', 'right', 'green')    
    # four_position('paligemma', 'neg_left', 'green')
    # four_position('paligemma', 'neg_right', 'green')
    # four_position('paligemma', 'left', 'yellow')
    # four_position('paligemma', 'right', 'yellow')
    # four_position('paligemma', 'neg_left', 'yellow')
    # four_position('paligemma', 'neg_right', 'yellow')