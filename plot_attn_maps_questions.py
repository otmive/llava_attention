
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration


def plot_attn_maps_questions(model_id, image_path):
    if model_id == "OpenGVLab/InternVL3_5-4B-HF":
        from plotter2 import Plotter
    else:
        print("Using standard plotter")
        from plotter import Plotter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global _loaded_models
    if '_loaded_models' not in globals():
        _loaded_models = {}

    if model_id not in _loaded_models:
        print("Loading model and processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        if "InternVL" in model_id:
            model = InternVLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16).to(device)
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16).to(device)
        _loaded_models[model_id] = (processor, model)
    else:
        print("Reusing cached model and processor.")
        processor, model = _loaded_models[model_id]


    plotter = Plotter(image_path)
    plotter.set_model(model, processor)
    # get colour of left shape
    left_colour = plotter.get_left_shapes()[0].colour
    right_colour = plotter.get_right_shapes()[0].colour
    plotter.get_outputs(f"Is the left shape {left_colour}? Answer with one word.")
    print("output:")
    print(plotter.print_output())
    if "llava" in model_id.lower():
        folder = "llava_question_plots"
    else:
        folder = "intervl_question_plots"
    for i in range(32):
        plotter.plot_image_attention(f"{folder}/{i}.png", i)

#plot_attn_maps_questions("OpenGVLab/InternVL3_5-4B-HF", "2d_dataset_fixed_positions_1000/images/image_0000.png")
plot_attn_maps_questions("llava-hf/llava-1.5-7b-hf", "2d_dataset_fixed_positions_1000/images/image_0000.png")
