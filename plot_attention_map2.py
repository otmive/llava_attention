from plotter2 import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoProcessor, InternVLForConditionalGeneration
# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
global _loaded_models
if '_loaded_models' not in globals():
    _loaded_models = {}
model_id = "OpenGVLab/InternVL3_5-4B-HF"
if model_id not in _loaded_models:
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = InternVLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        # device_map="auto",
        attn_implementation="eager",
        output_attentions=True
    ).to(device)
    _loaded_models[model_id] = (processor, model)
else:
    print("Reusing cached model and processor.")
    processor, model = _loaded_models[model_id]

image_path = '2d_dataset_fixed_positions_1000/images/image_0000.png'
plotter = Plotter(image_path)
plotter.set_model(model, processor)
# get colour of left shape
#left_colour = plotter.get_shape_by_position('top_left').colour
left_colour = plotter.get_left_shapes()[0].colour
# right_colour = plotter.get_right_shapes()[0].colour
plotter.get_outputs(f"The figure is {left_colour}")
print(plotter.print_output())


for n in range(36):
    plotter.plot_image_attention(f"internvl_test_2_shapes_layer{n}.png", n)


plotter.plot_attention_through_layers("internvl_layer_graph_image_0000.png")
# plotter.plot_image_attention("interntest_internvl.png")
# plotter.plot_bbox_on_attention_map("bbox_on_attention_map_internvl.png")