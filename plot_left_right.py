import torch
from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration
from transformers import AutoModelForImageTextToText
import gc
import multiprocessing as mp 

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

def plot_left_right(model_name, image_path):
    print("loading model ", model_name)
    processor, model = load_model(model_name)

    plotter = Plotter(image_path)
    plotter.set_model(model, processor)
    left_colour = plotter.get_left_shapes()[0].colour
    right_colour = plotter.get_right_shapes()[0].colour
    plotter.get_outputs(f"The figure is {left_colour}")
    l_bbox_attentions, l_baseline_attn = plotter.plot_attention_through_layers()
    plotter.get_outputs(f"The figure is {right_colour}")
    r_bbox_attentions, r_baseline_attn = plotter.plot_attention_through_layers()
    plotter.get_outputs(f"The figure is not {left_colour}")
    ln_bbox_attentions, ln_baseline_attn = plotter.plot_attention_through_layers()
    plotter.get_outputs(f"The figure is not{right_colour}")
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
    plt.plot(layers, l_bbox_attentions[left_colour], label=f'Left Shape Attention ({left_colour})', color= left_colour if left_colour in ['red', 'blue', 'green'] else 'gold')
    plt.plot(layers, l_bbox_attentions[right_colour], label=f'Right Shape Attention ({right_colour})', color=right_colour if right_colour in ['red', 'blue', 'green'] else 'gold')
    plt.plot(layers, l_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, overall_max * 1.1)  
    plt.savefig(f'{model_name}_left_{image_path.split("/")[-1].split(".")[0]}_attention.png')

    # plot the attentions for the right colour in a plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, r_bbox_attentions[left_colour], label=f'Left Shape Attention ({left_colour})', color=left_colour if left_colour in ['red', 'blue', 'green'] else 'gold')
    plt.plot(layers, r_bbox_attentions[right_colour], label=f'Right Shape Attention ({right_colour})', color=right_colour if right_colour in ['red', 'blue', 'green'] else 'gold')
    plt.plot(layers, r_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, overall_max * 1.1)  
    plt.savefig(f'{model_name}_right_{image_path.split("/")[-1].split(".")[0]}_attention.png')
    
    # plot negated versions
    plt.figure(figsize=(10, 6))
    plt.plot(layers, ln_bbox_attentions[left_colour], label=f'Left Shape Attention ({left_colour})', color= left_colour if left_colour in ['red', 'blue', 'green'] else 'gold')
    plt.plot(layers, ln_bbox_attentions[right_colour], label=f'Right Shape Attention ({right_colour})', color=right_colour if right_colour in ['red', 'blue', 'green'] else 'gold')
    plt.plot(layers, ln_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, overall_max * 1.1)  
    plt.savefig(f'{model_name}_neg_left_{image_path.split("/")[-1].split(".")[0]}_attention.png')

    # plot the attentions for the right colour in a plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, rn_bbox_attentions[left_colour], label=f'Left Shape Attention ({left_colour})', color=left_colour if left_colour in ['red', 'blue', 'green'] else 'gold')
    plt.plot(layers, rn_bbox_attentions[right_colour], label=f'Right Shape Attention ({right_colour})', color=right_colour if right_colour in ['red', 'blue', 'green'] else 'gold')
    plt.plot(layers, rn_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, overall_max * 1.1)  
    plt.savefig(f'{model_name}_neg_right_{image_path.split("/")[-1].split(".")[0]}_attention.png')
    # free up memory
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Finished plotting for ", model_name)

# image_path = "2d_dataset_fixed_positions_1000/images/image_0000.png"
# plot_left_right("llava", image_path)
# plot_left_right("internvl", image_path)
# plot_left_right("paligemma", image_path)

if __name__ == "__main__":
    image_path = "2d_dataset_fixed_positions_1000/images/image_0001.png"
    model_names = ["llava", "internvl", "paligemma"]
    processes = []
    for model_name in model_names:
        p = mp.Process(target=plot_left_right, args=(model_name, image_path))
        p.start()
        p.join()