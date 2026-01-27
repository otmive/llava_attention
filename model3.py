from plotter import Plotter

# load blip2 model and processor
from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import os
import gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print(torch.cuda.memory_summary(device=0, abbreviated=False))

HF_TOKEN = "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"

# load mistral model and processor
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"
global _loaded_models
if '_loaded_models' not in globals():
    _loaded_models = {}
# model_id = "Salesforce/mistral-7b-instruct-v0.2"
# if model_id not in _loaded_models:
#     print("Loading model and processor...")
#     processor = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
#     _loaded_models[model_id] = (processor, model)
# else:
#     print("Reusing cached model and processor.")
#     processor, model = _loaded_models[model_id]
# load blip vqa base
# model_id = "Salesforce/blip-image-captioning-base"
# if model_id not in _loaded_models:
#     print("Loading model and processor...")
#     processor = AutoProcessor.from_pretrained(model_id)
#     model = BlipForConditionalGeneration.from_pretrained(
#         model_id, torch_dtype=torch.float16).to(device)
#     _loaded_models[model_id] = (processor, model)
# else:
#     print("Reusing cached model and processor.")
#     processor, model = _loaded_models[model_id]

# load QWEN-VL
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
# if model_id not in _loaded_models:
#     print("Loading model and processor...")
#     processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

#     model = AutoModelForImageTextToText.from_pretrained(
#     model_id,
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="eager"  # MUST be 'eager' to see attentions
# )
#     _loaded_models[model_id] = (processor, model)
# else:
#     print("Reusing cached model and processor.")
#     processor, model = _loaded_models[model_id]

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

image_path = '2d_dataset_fixed_positions_1000/images/image_0000.png'



plotter = Plotter(image_path)
plotter.set_model(model, processor)
plotter.get_outputs("The figure is yellow")
print(plotter.print_output())
print(len(plotter.outputs["attentions"]))  
print(plotter.outputs["attentions"][0][0].shape)  # (batch_size, num_heads, seq_len, seq_len)
# print options for outputs
print("Output keys:", plotter.outputs.keys())
# print list of all tokens
tokens = [plotter.processor.decode(i) for i in plotter.outputs.sequences[0]]
print("Tokens:", tokens)
#plotter.plot_bbox_on_attention_map("test_qwen_map.png")
plotter.plot_image_attention(save_path="test_paligemma_image_attention.png")
plotter.plot_attention_through_layers(save_path='test_paligemma_layer_attention.png')