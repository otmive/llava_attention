from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration
from PIL import Image

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


processor, model = load_model("llava")
image_path = "2d_dataset_fixed_positions_1000/images/image_0000.png"
img = Image.open(image_path).convert("RGB")
print("image size:", img.size)
plotter = Plotter(image_path)
plotter.set_model(model, processor)
plotter.get_outputs("The figure is green")
print(plotter.print_output())
print(plotter.outputs["attentions"][0][0].shape) 
# left_colour = plotter.get_left_shapes()[0].colour
# right_colour = plotter.get_right_shapes()[0].colour
# plotter.set_model(model, processor)
# plotter.get_outputs(f"The figure is {left_colour}")
# plotter.plot_bbox_on_attention_map("test_llava_map.png")


