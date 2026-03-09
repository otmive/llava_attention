import json
from plotter import Plotter
from PIL import Image
import torch
from transformers import AutoProcessor
from transformers import InternVLForConditionalGeneration
from transformers import LlavaForConditionalGeneration, AutoModelForImageTextToText
import os
import torch 

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

with open('whatsup/detections.json', 'r') as file:
    data = json.load(file)


print(f"Total images with detections: {len(data)}")
print("first image with detections: ")

# print out first 5 entries
# for img_path, detections in list(data.items())[0]:

# for img_path in os.listdir('whatsup/controlled_images'):


# img_path = 'apple_left_of_armchair.jpeg'
# print(f"Image Path: {img_path}")

# processor, model = load_model("internvl")
# full_img_path = f"whatsup/images/{img_path}"
# plotter = Plotter(full_img_path)
# plotter.set_model(model, processor)
# left_shape = plotter.get_left_shapes()
# print("Left shape:", left_shape[0].shape)

# # print size of image
# image = Image.open(full_img_path)
# width, height = image.size
# print(f"Image size: {width}x{height}")
# # print memory usage at this point
# print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")    

# plotter.get_outputs("The object is a apple")
# print("ran get_outputs")
# # plotter.plot_bbox_on_attention_map(save_path="test_bbox_map.png")
# # print("plotted bbox on attention map")
# # plotter.plot_bbox_on_image(save_path="test_bbox_image.png")
# # print("plotted bbox on image")
# # plotter.plot_image_attention(save_path="test_image_attention2.png")
# plotter.plot_attention_through_layers(save_path="apple_chair_llava.png")
# print("plotted attention through layers")
# print(plotter.print_output())
processor, model = load_model("llava")

for image in os.listdir('whatsup/images'):
    plotter = Plotter(f"whatsup/images/{image}")
    plotter.set_model(model, processor)
    right_colour = plotter.get_right_shapes()[0].colour
    left_colour = plotter.get_left_shapes()[0].colour
    plotter.get_outputs(f"The object is not a {right_colour}")
    plotter.plot_attention_through_layers(save_path=f"plots/llava_whatsup_right_neg/{image}_{right_colour}.png")