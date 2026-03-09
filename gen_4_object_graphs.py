from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration, AutoModelForImageTextToText
import argparse

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
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.float16).to(device)
            _loaded_models[model_id] = (processor, model)
        else:
            print("Reusing cached model and processor.")
            processor, model = _loaded_models[model_id]


    return processor, model

def four_position(model_name, pos, ylim, colour=None):
   
    processor, model = load_model(model_name)

    count = 0
    item1_attn = []
    item2_attn = []
    item3_attn = []
    item4_attn = []
    baseline_attentions = []
    for i in range(5):
        image_path = f"4_shapes_same_dataset_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
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
       

        if target_colour == colour or colour is None:

            if pos == 'top_left':
                plotter.get_outputs(f"The figure is {top_left_colour}")
            elif pos == 'top_right':
                plotter.get_outputs(f"The figure is {top_right_colour}")
            elif pos == 'bottom_left':
                plotter.get_outputs(f"The figure is {bottom_left_colour}")
            elif pos == 'bottom_right':
                plotter.get_outputs(f"The figure is {bottom_right_colour}")
            elif pos == 'neg_top_left':
                plotter.get_outputs(f"The figure is not {top_left_colour}")
            elif pos == 'neg_top_right':
                plotter.get_outputs(f"The figure is not {top_right_colour}")
            elif pos == 'neg_bottom_left':
                plotter.get_outputs(f"The figure is not {bottom_left_colour}")
            elif pos == 'neg_bottom_right':
                plotter.get_outputs(f"The figure is not {bottom_right_colour}")

            print("outputs:", plotter.print_output())
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            item1_attn.append(bbox_attentions[top_left_colour])
            item2_attn.append(bbox_attentions[top_right_colour])
            item3_attn.append(bbox_attentions[bottom_left_colour])
            item4_attn.append(bbox_attentions[bottom_right_colour])
            baseline_attentions.append(baseline_attn)
            count += 1

    print("finished processing images, now plotting averages...")

    # plot average attention across items
    avg_item1_attn = np.mean(np.array(item1_attn), axis=0)
    avg_item2_attn = np.mean(np.array(item2_attn), axis=0)
    avg_item3_attn = np.mean(np.array(item3_attn), axis=0)
    avg_item4_attn = np.mean(np.array(item4_attn), axis=0)
    avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)
    std_item1_attn = np.std(np.array(item1_attn), axis=0)
    std_item2_attn = np.std(np.array(item2_attn), axis=0)
    std_item3_attn = np.std(np.array(item3_attn), axis=0)
    std_item4_attn = np.std(np.array(item4_attn), axis=0)
    if model_name == 'llava':
        layers = list(range(32))
    elif model_name == 'internvl':
        layers = list(range(36))
    elif model_name == 'paligemma':
        layers = list(range(26))
    plt.figure(figsize=(10, 6))
    # plt.plot(layers, avg_item1_attn, label='Top Left Shape Attention')
    # plt.plot(layers, avg_item2_attn, label='Top Right Shape Attention')
    # plt.plot(layers, avg_item3_attn, label='Bottom Left Shape Attention')
    # plt.plot(layers, avg_item4_attn, label='Bottom Right Shape Attention')

    plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    # plot mentioned shape and non-mentinoed shapes (averaged together)

    




    # plot confidence intervals
    # plt.fill_between(layers, avg_item1_attn - std_item1_attn, avg_item1_attn + std_item1_attn, color='blue', alpha=0.2)
    # plt.fill_between(layers, avg_item2_attn - std_item2_attn, avg_item2_attn + std_item2_attn, color='red', alpha=0.2)    
    # plt.fill_between(layers, avg_item3_attn - std_item3_attn, avg_item3_attn + std_item3_attn, color='green', alpha=0.2)
    # plt.fill_between(layers, avg_item4_attn - std_item4_attn, avg_item4_attn + std_item4_attn, color='orange', alpha=0.2)
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Average Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, ylim)
    #plt.ylim(0, 0.01)
    #plt.ylim(0, 0.02)
    plt.savefig(f'plots/4_obj_plots/{model_name}_{pos}_{colour+"_" if colour else ""}all.png')
    print(f"Processed {count} images matching criteria.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llava', help='Model name: llava, internvl, paligemma')
    parser.add_argument('--ylim', type=float, default=None, help='Y-axis limits for the plot')
    args = parser.parse_args()
    four_position(args.model_name, 'top_left', args.ylim)
    four_position(args.model_name, 'top_right', args.ylim)
    four_position(args.model_name, 'bottom_left', args.ylim)
    four_position(args.model_name, 'bottom_right', args.ylim)
    four_position(args.model_name, 'neg_top_left', args.ylim)
    four_position(args.model_name, 'neg_top_right', args.ylim)
    four_position(args.model_name, 'neg_bottom_left', args.ylim)
    four_position(args.model_name, 'neg_bottom_right', args.ylim)
    # four_position('internvl', 'top_left', 'red')
    # four_position('internvl', 'top_right', 'red')
    # four_position('internvl', 'bottom_left', 'red')
    # four_position('internvl', 'bottom_right', 'red')
    # four_position('internvl', 'neg_top_left', 'red')
    # four_position('internvl', 'neg_top_right', 'red')
    # four_position('internvl', 'neg_bottom_left', 'red')
    # four_position('internvl', 'neg_bottom_right', 'red')
