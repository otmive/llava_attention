from plotter import Plotter
import torch
from transformers import AutoProcessor, InternVLForConditionalGeneration, AutoModelForImageTextToText, LlavaForConditionalGeneration
import os

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

def plot_attention_heatmap(model_name, image_path):

    processor, model = load_model(model_name)
    plotter = Plotter(image_path)
    left_colour = plotter.get_left_shapes()[0].colour
    right_colour = plotter.get_right_shapes()[0].colour
    plotter.set_model(model, processor)
    plotter.get_outputs(f"The figure is {left_colour}")
    basename = os.path.splitext(os.path.basename(image_path))[0]
    for l in range(len(plotter.outputs.attentions[0])):
        plotter.plot_image_attention(f"plots/heatmaps/{model_name}_{basename}_{l}", l)

if __name__ == "__main__":
    #plot_attention_heatmap("llava", "whatsup/images/apple_left_of_armchair.jpeg")
    plot_attention_heatmap("internvl", "whatsup/images/book_right_of_table.jpeg")
    #plot_attention_heatmap("paligemma", "whatsup/images/book_right_of_table.jpeg")