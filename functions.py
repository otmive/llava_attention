import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import InternVLForConditionalGeneration, LlavaForConditionalGeneration
import numpy as np


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
                model_id, torch_dtype=torch.float16, attn_implementation="eager").to(device)
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
                model_id, torch_dtype=torch.float16, attn_implementation="eager").to(device)
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
                model_id, torch_dtype=torch.float16, attn_implementation="eager").to(device)
            _loaded_models[model_id] = (processor, model)
        else:
            print("Reusing cached model and processor.")
            processor, model = _loaded_models[model_id]


    return processor, model

def mean_and_ci(arr):
  CI_Z = 1.96 # 95%
  n = len(arr)
  se = np.std(arr, ddof=1) / np.sqrt(n)
  return arr.mean(), CI_Z * se