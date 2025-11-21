from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from PIL import Image
import requests
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import math
import gc
import torch
import os
import torch.nn.functional as F
import json
from functions import vis_attention_head_layer, get_outputs, get_matrix_layer_all_heads, get_image_attention, get_bbox, get_bbox_attention
torch.cuda.empty_cache()
gc.collect()

def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])

def aggregate_llm_attention_layer(attn, layer_idx):
    '''Extract average attention vector for a specific layer'''
    layer = attn[layer_idx]
    layer_attns = layer.squeeze(0)
    attns_per_head = layer_attns.mean(dim=0)
    vec = torch.concat((
        # We zero the first entry because it's what's called
        # null attention (https://aclanthology.org/W19-4808.pdf)
        torch.tensor([0.]),
        # usually there's only one item in attns_per_head but
        # on the first generation, there's a row for each token
        # in the prompt as well, so take [-1]
        attns_per_head[-1][1:].cpu(),
        # add zero for the final generated token, which never
        # gets any attention
        torch.tensor([0.]),
    ))
    return vec / vec.sum()

def get_image_attention_for_layer(outputs, processor, inputs, layer_idx, image_url, colour):

    input_token_len = len(inputs["input_ids"][0])
    tokens = [processor.tokenizer.decode(i) for i in outputs.sequences[0]]
    attention_matrix = get_matrix_layer_all_heads(outputs, layer_idx)

    image_tok = '<image>'
    # get attention from output to image tokens
    image_indices = [i for i, token in enumerate(tokens) if token == image_tok]



    attn_out_to_image = get_image_attention(attention_matrix, inputs, outputs, processor)



    # plot bbox over image
    image = Image.open(image_url)
    image_np = np.array(image) / 255.0
    H, W, _ = image_np.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np*0.4)
    # add bbox
    bbox = get_bbox(image_url, colour)
    if bbox:
        x1, y1, x2, y2 = bbox
        print(bbox)
        print("BBOX^^")


        # convert to integers
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    print(f"Image size: W={W}, H={H}")


    attn_tensor = torch.tensor(attn_out_to_image)
    # create attention heatmap to overlay on image
    #attn_out_to_image = Image.fromarray((attn_out_to_image * 255).astype(np.uint8))
    attn_over_image = F.interpolate(
    attn_tensor.unsqueeze(0).unsqueeze(0), 
    size=image.size[::-1],  # PIL uses (W, H), PyTorch uses (H, W)
    mode='bicubic', 
    align_corners=False
    ).squeeze()
    #attn_out_to_image = np.array(attn_out_to_image) / 255.0
    #ax.imshow(attn_over_image, cmap='viridis', alpha=0.7)
    ax.set_title(f"Layer {layer_idx}: Attention overlay on image")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # scale bounding box to attention heatmap size
    scale = attn_out_to_image.shape[0] / W
    print(scale)
    bbox_heatmap = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
    # plot bounding box over attention heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(attn_out_to_image, cmap='viridis', alpha=0.7)
    x1, y1, x2, y2 = bbox_heatmap
    print(bbox_heatmap)
    print(attn_out_to_image.shape)
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"Layer {layer_idx}: Attention heatmap with bounding box")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # get attention within the bounding box
    x1, y1, x2, y2 = bbox_heatmap
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    print(f"Bounding box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    
    # plot attention heatmap but only for values within the bounding box
    attn_within_bbox = attn_out_to_image[y1:y2+1, x1:x2+1]
    print("attention within bbox", attn_within_bbox.sum())
    print("new attention within bbox", get_bbox_attention(bbox, attn_out_to_image))
    
    # edit heatmap so all bounding box values are set to 1
    attn_mask = np.zeros_like(attn_out_to_image)
    attn_mask[y1:y2+1, x1:x2+1] = 1.0
    # plot attn_mask
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(attn_mask, cmap='viridis', alpha=0.7)
    ax.set_title(f"Layer {layer_idx}: Attention mask within bounding box")
    ax.axis('off')
    plt.tight_layout()  
    plt.show()

    # plot attention mask overlayed onto image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)
    attn_mask_resized = torch.tensor(attn_mask)
    # resize attention mask to image size
    attn_mask_resized = F.interpolate(
        torch.tensor(attn_mask).unsqueeze(0).unsqueeze(0),
        size=image.size[::-1],  # PIL uses (W, H), PyTorch uses (H, W)
        mode='bicubic',
        align_corners=False
    ).squeeze().numpy()
    ax.imshow(attn_mask_resized, cmap='viridis', alpha=0.7  )
    ax.set_title(f"Layer {layer_idx}: Attention mask overlay on image")
    ax.axis('off')
    plt.tight_layout()
    
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global _loaded_models
    if '_loaded_models' not in globals():
        _loaded_models = {}

    model_id = "llava-hf/llava-1.5-7b-hf"
    if model_id not in _loaded_models:
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
        _loaded_models[model_id] = (processor, model)
    else:
        print("Reusing cached model and processor.")
        processor, model = _loaded_models[model_id]

    return model, processor

def vis_attention(image_url, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    load_model()
    # Create directories for outputs
    os.makedirs("Figures", exist_ok=True)
    os.makedirs("Figures/layers", exist_ok=True)
    os.makedirs("Figures/layer_attention_maps", exist_ok=True)

    
    model, processor = load_model()
    


    image = Image.open(image_url)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process the image
    processed = processor(images=image, text="", return_tensors="pt")
    print("Processed pixel_values shape:", processed["pixel_values"].shape)

    inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
    print(len(inputs))

    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, output_attentions=True, return_dict_in_generate=True)
    print(outputs[0])
    print("OUTPUTS")
    model_answer = processor.decode(outputs[0][0], skip_special_tokens=True)
    print(len(outputs['attentions']))
    tokens = [processor.tokenizer.decode(i) for i in outputs.sequences[0]]
    print(tokens)

    image_tok = '<image>'
    num_layers = len(outputs["attentions"][0])  # Get number of layers

    # Process attention for each layer separately
    aggregated_prompt_attention_per_layer = []
    for layer_idx in range(num_layers):
        layer_attention = []
        for i, layer in enumerate(outputs["attentions"]):
            if i == 0:  # First output token
                layer_attns = layer[layer_idx].squeeze(0)
                attns_per_head = layer_attns.mean(dim=0)
                cur = attns_per_head[:-1].cpu().clone()
                # Zero out attention to first <bos> token for non-first positions
                cur[1:, 0] = 0.
                cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
                layer_attention.append(cur)
                break
        aggregated_prompt_attention_per_layer.append(torch.stack(layer_attention).mean(dim=0))

    # Create attention matrices for each layer
    layer_attention_matrices = []
    for layer_idx in range(num_layers):
        # Get layer-specific attention weights for generated tokens
        layer_gen_attentions = []
        for output_attn in outputs["attentions"]:
            layer_gen_attentions.append(aggregate_llm_attention_layer(output_attn, layer_idx))
        
        # Build attention matrix for this layer
        llm_attn_matrix_layer = heterogenous_stack(
            [torch.tensor([1])]
            + list(aggregated_prompt_attention_per_layer[0])  # Use first layer's prompt attention as baseline
            + layer_gen_attentions
        )
        
        layer_attention_matrices.append(llm_attn_matrix_layer)
        print(f"Layer {layer_idx} attention matrix shape: {llm_attn_matrix_layer.shape}")

    # Plot attention matrix for each layer
    gamma_factor = 1
    for layer_idx, attn_matrix in enumerate(layer_attention_matrices):
        enhanced_attn_m = np.power(attn_matrix.numpy(), 1 / gamma_factor)
        
        fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
        ax.imshow(enhanced_attn_m, vmin=enhanced_attn_m.min(), vmax=enhanced_attn_m.max(), interpolation="nearest")
        ax.set_title(f"Attention Matrix - Layer {layer_idx}")
        plt.savefig(f"Figures/layers/matrix_layer_{layer_idx:02d}.png")
        plt.close()

    input_token_len = len(inputs["input_ids"][0])
    output_token_len = len(tokens) - input_token_len

    image_indices = [i for i, token in enumerate(tokens) if token == image_tok]
    image_token_start = image_indices[0]
    image_token_end = image_indices[-1]
    image_token_indices = image_indices
    grid_size = int(len(image_token_indices) ** 0.5)

    # Process attention maps for each layer
    image_np = np.array(image) / 255.0
    H, W, _ = image_np.shape
    
    for layer_idx, attn_matrix in enumerate(layer_attention_matrices):
        # Calculate attention weights over vision tokens for this layer
        overall_attn_weights_over_vis_tokens = []
        for i, row in enumerate(attn_matrix[input_token_len:]):
            overall_attn_weights_over_vis_tokens.append(
                row[image_token_start:image_token_end].sum().item()
            )

        # Plot attention weights over vision tokens for this layer
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(overall_attn_weights_over_vis_tokens)
        ax.set_xticks(range(len(overall_attn_weights_over_vis_tokens)))
        ax.set_xticklabels(tokens[-output_token_len:], rotation=75)
        ax.set_title(f"Layer {layer_idx}: Sum of attention weights over vision tokens")
        plt.tight_layout()
        plt.savefig(f"Figures/layers/attention_over_vis_tokens_layer_{layer_idx:02d}.png")
        plt.close()

        # Calculate attention to image tokens for each output token in this layer
        output_token_indices = list(range(len(tokens) - output_token_len, len(tokens)))
        output_tokens = tokens[-output_token_len:]
        
        all_output_attentions = []
        for i, (out_idx, token_label) in enumerate(zip(output_token_indices, output_tokens)):
            image_attn = attn_matrix[out_idx, image_token_indices]
            image_attn_grid = image_attn.reshape(grid_size, grid_size)
            all_output_attentions.append(image_attn_grid.cpu().numpy())

        # Average attention for this layer
        avg_attn = np.mean(all_output_attentions, axis=0)
        
        # Plot attention heatmap for this layer
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(avg_attn, cmap="viridis")
        ax.set_title(f"Layer {layer_idx}: Average attention to image tokens")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"Figures/layer_attention_maps/attention_heatmap_layer_{layer_idx:02d}.png")
        plt.close()

        # Plot attention overlaid on image for this layer
        avg_attn_img = Image.fromarray((avg_attn * 255).astype(np.uint8))
        avg_attn_img = avg_attn_img.resize((W, H), resample=Image.BICUBIC)
        avg_attn_img = np.array(avg_attn_img) / 255.0

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np)
        ax.imshow(avg_attn_img, cmap='viridis', alpha=0.5)
        ax.set_title(f"Layer {layer_idx}: Attention overlay on image")
        ax.axis('off')
        plt.tight_layout()
        print("PLOTTNG OVERLAY")
        plt.savefig(f"Figures_4/layer_attention_maps/attention_overlay_layer_{layer_idx:02d}.png")
        plt.close()

    print(f"\nGenerated attention visualizations for {num_layers} layers")
    print("Files saved in:")
    print("- Figures/layers/ (attention matrices and token attention plots)")
    print("- Figures/layer_attention_maps/ (attention heatmaps and overlays)")

if __name__ == "__main__":
    # vis_attention("2d_dataset_fixed_positions_1000/images/image_0007.png", "The figure")
    # outputs, processor, inputs = get_outputs("4_shapes_dataset_100/images/image_0000.png", f"The figure is red.")
    # vis_attention_head_layer(outputs, processor, inputs, "4_shapes_dataset_100/images/image_0000.png")

    # for i in range(32):
    #     get_image_attention_for_layer(outputs, processor, inputs, i, "4_shapes_dataset_100/images/image_0000.png", "red")

    vis_attention("4_shapes_dataset_100/images/image_0006.png", "What colour are the shapes on the bottom?")