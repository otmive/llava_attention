from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVision2Seq
# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
global _loaded_models
if '_loaded_models' not in globals():
    _loaded_models = {}
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
#model_id = "llava-hf/llava-1.5-7b-hf"
if model_id not in _loaded_models:
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    if "Qwen" in model_id:
        model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16).to(device)
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
    _loaded_models[model_id] = (processor, model)
else:
    print("Reusing cached model and processor.")
    processor, model = _loaded_models[model_id]

image_path = "2d_dataset_fixed_positions_1000/images/image_0000.png"
plotter = Plotter(image_path)
plotter.set_model(model, processor)
# get colour of left shape
left_colour = plotter.get_left_shapes()[0].colour
right_colour = plotter.get_right_shapes()[0].colour
plotter.get_outputs(f"The figure is {left_colour}")

plotter.plot_image_attention("qwen_test_attention_map.png")