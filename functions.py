import torch
import os
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt
import json

_loaded_models = {}

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

def vis_attention_layers(image_url, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create directories for outputs
    os.makedirs("Figures", exist_ok=True)
    os.makedirs("Figures/layers", exist_ok=True)
    os.makedirs("Figures/layer_attention_maps", exist_ok=True)

    model_id = "llava-hf/llava-1.5-7b-hf"

    # Initialize _loaded_models if not exists
    global _loaded_models
    if '_loaded_models' not in globals():
        _loaded_models = {}

    if model_id not in _loaded_models:
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
        _loaded_models[model_id] = (processor, model)
    else:
        print("Reusing cached model and processor.")
        processor, model = _loaded_models[model_id]

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
    print(processor.decode(outputs[0][0], skip_special_tokens=True))
    print(len(outputs['attentions']))
    tokens = [processor.tokenizer.decode(i) for i in outputs.sequences[0]]
    print(tokens)

    image_tok = '<image>'
    num_layers = len(outputs["attentions"][0])  # Get number of layers
    print(f"Number of layers: {num_layers}")

    # Process prompt attention for each layer separately
    aggregated_prompt_attention_per_layer = []
    for layer_idx in range(num_layers):
        # Get the attention from the first output token for this specific layer
        first_output_attn = outputs["attentions"][0][layer_idx]  # First output token, specific layer
        layer_attns = first_output_attn.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        # Zero out attention to first <bos> token for non-first positions
        cur[1:, 0] = 0.
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention_per_layer.append(cur)

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
            + list(aggregated_prompt_attention_per_layer[layer_idx])  # Use layer-specific prompt attention
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
        plt.savefig(f"Figures/layer_attention_maps/attention_overlay_layer_{layer_idx:02d}.png")
        plt.close()

    print(f"\nGenerated attention visualizations for {num_layers} layers")
    print("Files saved in:")
    print("- Figures/layers/ (attention matrices and token attention plots)")
    print("- Figures/layer_attention_maps/ (attention heatmaps and overlays)")


def vis_attention_head_layer_save(image_url, text_prompt, head_idx, layer_idx):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("I AM IN HERE")
    # Create directories for outputs
    os.makedirs("Figures", exist_ok=True)
    os.makedirs("Figures/layers", exist_ok=True)
    os.makedirs("Figures/layer_attention_maps", exist_ok=True)

    model_id = "llava-hf/llava-1.5-7b-hf"

    # Initialize _loaded_models if not exists
    global _loaded_models
    if '_loaded_models' not in globals():
        _loaded_models = {}

    if model_id not in _loaded_models:
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
        _loaded_models[model_id] = (processor, model)
    else:
        print("Reusing cached model and processor.")
        processor, model = _loaded_models[model_id]

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
    print(processor.decode(outputs[0][0], skip_special_tokens=True))
    print(len(outputs['attentions']))
    tokens = [processor.tokenizer.decode(i) for i in outputs.sequences[0]]
    print(tokens)

    image_tok = '<image>'
    num_layers = len(outputs["attentions"][0])  # Get number of layers
    print(f"Number of layers: {num_layers}")

    # get attentin for spedcified head and layer
    layer_attentions = outputs["attentions"][0][layer_idx].squeeze(0)
    print("layer shape", layer_attentions.shape)
    head_attention = layer_attentions[head_idx]
    print("head attention shape", head_attention.shape)

    # plot attention for the specified head
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(head_attention.cpu().numpy(), cmap='viridis', interpolation='nearest')
    ax.set_title(f"Attention Head {head_idx} - Layer {layer_idx}")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"Figures/layers/attention_head_{head_idx}_layer_{layer_idx}.png")
    plt.close()

def get_outputs(image_url, text_prompt, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
    """Get model outputs for a given image and text prompt."""
    global _loaded_models
    _loaded_models = {}
    if model_id not in _loaded_models:
        print(f"Loading model: {model_id} onto {device}...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        _loaded_models[model_id] = (processor, model)
    else:
        processor, model = _loaded_models[model_id]
    # processor = AutoProcessor.from_pretrained(model_id)
    # model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

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

    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, output_attentions=True, return_dict_in_generate=True)

    return outputs, processor, inputs

def get_outputs_flipped(image_url, text_prompt, model_id="llava-hf/llava-1.5-7b-hf", device="cuda", flip_positions=False):
    """Get model outputs with optional position embedding flipping for attention analysis."""
    global _loaded_models
    if model_id not in _loaded_models:
        print(f"Loading model: {model_id} onto {device}...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        _loaded_models[model_id] = (processor, model)
    else:
        processor, model = _loaded_models[model_id]

    vit = model.vision_tower.vision_model
    
    # Store original position embeddings
    original_pos_emb = vit.embeddings.position_embedding.weight.data.clone()
    
    if flip_positions:
        pos_emb = original_pos_emb.clone()  # Work with a copy
        cls_emb = pos_emb[0:1, :]          # class token
        patch_emb = pos_emb[1:, :]         # patch tokens
        grid_size = int(patch_emb.shape[0] ** 0.5)  # 24 for 24x24
        
        # Reshape to 2D grid
        patch_emb_2d = patch_emb.reshape(grid_size, grid_size, -1)
        
        # Flip horizontally (swap left/right)
        patch_emb_2d_flipped = torch.flip(patch_emb_2d, dims=[1])
        
        # Flatten back
        patch_emb_flipped = patch_emb_2d_flipped.reshape(-1, patch_emb_2d.shape[-1])
        
        # Concatenate class token and update
        vit.embeddings.position_embedding.weight.data = torch.cat([cls_emb, patch_emb_flipped], dim=0)

    try:
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
        
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False, 
            output_attentions=True, 
            return_dict_in_generate=True
        )
        
        return outputs, processor, inputs
    
    finally:
        # Always restore original embeddings
        vit.embeddings.position_embedding.weight.data = original_pos_emb

def get_outputs_with_token_reordering(image_url, text_prompt, model_id="llava-hf/llava-1.5-7b-hf", device="cuda", flip_token_order=False):
    """Flip the actual order of vision tokens fed to the language model."""
    global _loaded_models
    processor, model = _loaded_models[model_id]
    
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
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)
    
    if flip_token_order:
        # Hook into the model to flip vision token order
        original_forward = model.vision_tower.forward
        
        def flipped_forward(pixel_values, **kwargs):
            outputs = original_forward(pixel_values, **kwargs)
            
            # Get the vision features [batch, num_tokens, hidden_dim]
            vision_features = outputs.last_hidden_state
            
            cls_token = vision_features[:, 0:1, :]  # Keep CLS token
            patch_tokens = vision_features[:, 1:, :]  # Get patch tokens
            
            # Reshape to 2D grid
            batch_size = patch_tokens.shape[0]
            num_patches = patch_tokens.shape[1]
            grid_size = int(num_patches ** 0.5)
            hidden_dim = patch_tokens.shape[2]
            
            patch_grid = patch_tokens.reshape(batch_size, grid_size, grid_size, hidden_dim)
            
            # Flip horizontally
            patch_grid_flipped = torch.flip(patch_grid, dims=[2])  # Flip along width
            
            # Flatten back to sequence
            patch_tokens_flipped = patch_grid_flipped.reshape(batch_size, -1, hidden_dim)
            
            # Recombine with CLS token
            vision_features_flipped = torch.cat([cls_token, patch_tokens_flipped], dim=1)
            
            # Update the outputs
            outputs.last_hidden_state = vision_features_flipped
            return outputs
        
        model.vision_tower.forward = flipped_forward
    
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True
        )
        return outputs, processor, inputs
    finally:
        if flip_token_order:
            model.vision_tower.forward = original_forward

def plot_image_attention(image, attention_matrix, output_token_len, tokens, save_path=None, title=None):

    image_tok = '<image>'
    # get attention from output to image tokens
    output_indices = list(range(len(tokens) - output_token_len, len(tokens)))
    image_indices = [i for i, token in enumerate(tokens) if token == image_tok]
    image_token_start = image_indices[0]
    image_token_end = image_indices[-1]


    attn_out_to_image = attention_matrix[output_indices][:,image_indices]


    # average over all output tokens
    attn_out_to_image = attn_out_to_image.mean(dim=0)
    # resize to image grid
    grid_size = int(len(image_indices) ** 0.5)
    attn_out_to_image = attn_out_to_image.reshape(grid_size, grid_size).cpu().numpy()

    # plot attention heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(attn_out_to_image, cmap="viridis")
    ax.set_title(title if title else "Attention Heatmap")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def get_matrix_layer_all_heads(outputs, layer_idx):

    layer_attns = outputs["attentions"][0][layer_idx].squeeze(0)
    attns_per_head = layer_attns.mean(dim=0)
    cur = attns_per_head[:-1].cpu().clone()
    # Zero out attention to first <bos> token for non-first positions
    cur[1:, 0] = 0.
    # Normalize attention weights
    cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
    input_attentions = cur


    # get attention from output tokens to rest of tokens
    output_attentions = []
    for out in outputs["attentions"]:   
        layer_attns = out[layer_idx].squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        # zero attention to first <BOS> token and zero attention to final token
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
        # normalise
        output_attentions.append(vec / vec.sum())


    return heterogenous_stack(
        [torch.tensor([1])]  # Start with a dummy token
        + list(input_attentions)  # Add input attention
        + output_attentions  # Add output attentions
    )

def get_matrix_for_head_and_layer(outputs, layer_idx, head_idx):

    """Get attention matrix for a specific head in a specific layer"""
    layer_attns = outputs["attentions"][0][layer_idx].squeeze(0)
    attns_per_head = layer_attns[head_idx]  # Get specific head
    cur = attns_per_head[:-1].cpu().clone()
    # Zero out attention to first <bos> token for non-first positions
    cur[1:, 0] = 0.
    cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
    
    input_attentions = cur

    # get attention from output tokens to rest of tokens
    output_attentions = []
    for out in outputs["attentions"]:
        layer_attns = out[layer_idx].squeeze(0)
        attns_per_head = layer_attns[head_idx]  # Get specific head

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
        # normalise
        output_attentions.append(vec / vec.sum())

    return heterogenous_stack(
        [torch.tensor([1])]  # Start with a dummy token
        + list(input_attentions)  # Add input attention
        + output_attentions  # Add output attentions
    )

def get_matrix_all_layers_heads(outputs):

    num_layers = len(outputs["attentions"][0])  # Get number of layers

    aggregated_attn_all_layers_heads = []
    for l in range(num_layers):
        layer_attns = outputs["attentions"][0][l].squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        # Zero out attention to first <bos> token for non-first positions
        cur[1:, 0] = 0.
        # Normalize attention weights
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_attn_all_layers_heads.append(cur)
    
    # mean to get average across all layers
    aggregated_attn_all_layers_heads = torch.stack(aggregated_attn_all_layers_heads).mean(dim=0)

    # get attention from output tokens to rest of tokens
    output_attentions = []
    for out in outputs["attentions"]:
        avged = []
        for layer in out:
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            # zero attention to first <BOS> token and zero attention to final token
            vec = torch.concat((
                # We zero the first entry because it's what's called
                # null attention (https://aclanthology.org/W19-4808.pdf)
                torch.tensor([0.]),
                # usually there's only one item in attns_per_head but
                # on the first generation, there's a row for each token
                # in the prompt as well, so take [-1]
                attns_per_head[-1][1:].cpu(),
                # attns_per_head[-1].cpu(),
                # add zero for the final generated token, which never
                # gets any attention
                torch.tensor([0.]),
            ))
            # normalise
            avged.append(vec / vec.sum())
        output_attentions.append(torch.stack(avged).mean(dim=0))
    
    # create final attention matrix with padded zeros
    return heterogenous_stack(
    [torch.tensor([1])]
    + list(aggregated_attn_all_layers_heads) 
    + list(output_attentions))

# def heterogenous_stack(vecs):
#     '''Pad vectors with zeros then stack'''
#     max_length = max(v.shape[0] for v in vecs)
#     return torch.stack([
#         torch.concat((v, torch.zeros(max_length - v.shape[0])))
#         for v in vecs
#     ])

def vis_attention_head_layer(outputs, processor, inputs, image_url, layer_idx=None, head_idx=None):

    tokens = [processor.tokenizer.decode(i) for i in outputs.sequences[0]]


    if head_idx is None and layer_idx is None:
        llm_attn_matrix = get_matrix_all_layers_heads(outputs)
    elif head_idx is None and layer_idx is not None:
        llm_attn_matrix = get_matrix_layer_all_heads(outputs, layer_idx)
    else:
        llm_attn_matrix = get_matrix_for_head_and_layer(outputs, layer_idx, head_idx)

    print(f"Attention matrix shape: {llm_attn_matrix.shape}")
    # plot matrix
    gamma_factor = 1
    enhanced_attn_m = np.power(llm_attn_matrix.numpy(), 1 / gamma_factor)
    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
    ax.imshow(enhanced_attn_m, vmin=enhanced_attn_m.min(), vmax=enhanced_attn_m.max(), interpolation="nearest")
    ax.set_title(f"Attention Matrix - Layer {layer_idx}, Head {head_idx}")
    # plt.show()
    
    # get length of output tokens
    input_token_len = len(inputs["input_ids"][0])
    output_token_len = len(tokens) - input_token_len
    image_name = os.path.basename(image_url)
    image_name = os.path.splitext(image_name)[0]

    image = Image.open(image_url)

    plot_image_attention(image, llm_attn_matrix, output_token_len, tokens, save_path=f"attention_maps/attention_{image_name}_layer_{layer_idx}_head_{head_idx}.png", title=f"Attention Heatmap - Layer {layer_idx}, Head {head_idx}")
    
def get_bbox(image_url, colour):

    data_file = "/mnt/shared/home/ep16475/llava_attention/4_shapes_dataset_100/labels.json"
    W = 177
    with open(data_file, 'r') as f:
        data = json.load(f)

    print(image_url)
    image_name = os.path.basename(image_url)
    print(image_name)
    print(colour)
    for item in data:
        if item['filename'] == image_name:
            for pair in item['shape_color_pairs']:
                if pair['color']==colour:
                    bbox = pair['bbox']
                    
    orig_W, orig_H = 256, 256

    scale_x = W / orig_W
    scale_y = W / orig_H

    x1, y1, x2, y2 = bbox
    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y
    # flip coordinates in y axis
    y1 = W - y1
    y2 = W - y2

    # return swapped ys because y1 is bigger than y2
    return [x1, y2, x2, y1]

def get_image_attention(matrix, inputs, outputs, processor):

    input_token_len = len(inputs["input_ids"][0])
    output_token_len = len(outputs.sequences[0]) - input_token_len
    tokens = [processor.tokenizer.decode(i) for i in outputs.sequences[0]]
    image_tok = '<image>'
    # get attention from output to image tokens
    output_indices = list(range(len(tokens) - output_token_len, len(tokens)))
    image_indices = [i for i, token in enumerate(tokens) if token == image_tok]

    attn_out_to_image = matrix[output_indices][:,image_indices]

    # average over all output tokens
    attn_out_to_image = attn_out_to_image.mean(dim=0)
    # resize to image grid
    grid_size = int(len(image_indices) ** 0.5)
    return attn_out_to_image.reshape(grid_size, grid_size).cpu().numpy()


def get_bbox_attention(bbox, attn):


    x1, y1, x2, y2 = bbox

    # scale bounding box to attention heatmap size
    scale = attn.shape[0] / 177
    bbox_heatmap = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
    x1, y1, x2, y2 = bbox_heatmap

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # plot attention heatmap but only for values within the bounding box
    return attn[y1:y2+1, x1:x2+1].sum().item()

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
    plt.show()

def get_bbox_attention_for_layer(outputs, processor, inputs, layer_idx, image_url, colour):

    attention_matrix = get_matrix_layer_all_heads(outputs, layer_idx)

    attn_out_to_image = get_image_attention(attention_matrix, inputs, outputs, processor)

    bbox_attn = get_bbox_attention(get_bbox(image_url, colour), attn_out_to_image)

    return bbox_attn

def get_bbox_attention_token(bbox, attn):


    x1, y1, x2, y2 = bbox

    # scale bounding box to attention heatmap size
    scale = attn.shape[0] / 177
    bbox_heatmap = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
    x1, y1, x2, y2 = bbox_heatmap

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # calculate number of tokens in the bounding box

    bbox_tokens = (x2 - x1 + 1) * (y2 - y1 + 1)
    print("number tokens in bbox", bbox_tokens)
    # plot attention heatmap but only for values within the bounding box
    return attn[y1:y2+1, x1:x2+1].sum().item()/bbox_tokens

def get_bbox_attention_for_layer(outputs, processor, inputs, layer_idx, image_url, colour):

    attention_matrix = get_matrix_layer_all_heads(outputs, layer_idx)

    attn_out_to_image = get_image_attention(attention_matrix, inputs, outputs, processor)

    bbox_attn = get_bbox_attention_token(get_bbox(image_url, colour), attn_out_to_image)

    return bbox_attn

def get_bbox_attention_for_layer_4(outputs, processor, inputs, layer_idx, image_url, bbox):

    attention_matrix = get_matrix_layer_all_heads(outputs, layer_idx)

    attn_out_to_image = get_image_attention(attention_matrix, inputs, outputs, processor)

    bbox_attn = get_bbox_attention_token(bbox, attn_out_to_image)

    return bbox_attn

def get_baseline_attn(bbox1, bbox2, outputs, processor, inputs):

    base_attns = []
    for layer_idx in range(32):
        attention_matrix = get_matrix_layer_all_heads(outputs, layer_idx)

        attn_out_to_image = get_image_attention(attention_matrix, inputs, outputs, processor)

        x1_1, y1_1, x2_1, y2_1 = bbox1
        scale = attn_out_to_image.shape[0] / 177
        bbox1_heatmap = [x1_1 * scale, y1_1 * scale, x2_1 * scale, y2_1 * scale]
        x1_1, y1_1, x2_1, y2_1 = [int(x) for x in bbox1_heatmap]
        x1_2, y1_2, x2_2, y2_2 = bbox2
        bbox2_heatmap = [x1_2 * scale, y1_2 * scale, x2_2 * scale, y2_2 * scale]
        x1_2, y1_2, x2_2, y2_2 = [int(x) for x in bbox2_heatmap]

        # add the extra included tokens
        x2_1 += 1
        y2_1 += 1
        x2_2 += 1
        y2_2 += 1

        tok_count = 0
        for i in range(len(attn_out_to_image)):
            for j in range(len(attn_out_to_image[i])):
                # if token is within bbox1 or bbox2, set to 0
                if (x1_1 <= i <= x2_1 and y1_1 <= j <= y2_1) or (x1_2 <= i <= x2_2 and y1_2 <= j <= y2_2):
                    attn_out_to_image[i][j] = 0.0
                else:
                    tok_count += 1
        print("number of tokens outside bboxes", tok_count)
        base_attns.append(attn_out_to_image.sum().item()/ tok_count)

    return base_attns

def plot_attention_through_layers(image_url, position):
    # Get colours of objects in the image
    print("grabbing image " + image_url)
    image_data = "2d_dataset_vertical_positions/labels.json"
    with open(image_data, 'r') as f:
        data = json.load(f)

    image_name = os.path.basename(image_url)
    for item in data:
        if item['filename'] == image_name:
            colours = [pair['color'] for pair in item['shape_color_pairs']]
            bboxes = [pair['bbox'] for pair in item['shape_color_pairs']]

    # Get attention for each colour through layers
    if position == "left":
        col = colours[0]
    else:
        col = colours[1]
    outputs, processor, inputs = get_outputs(image_url, f"The figure is not {col}")
    layers = [i for i in range(32)]
    bbox_attn_per_layer = {colour: [] for colour in colours}
    for colour in colours:
        bbox_attn_per_layer[colour] = [get_bbox_attention_for_layer(outputs, processor, inputs, i, image_url, colour) for i in layers]
    
    baseline_attn = get_baseline_attn(get_bbox(image_url, colours[0]), get_bbox(image_url, colours[1]), outputs, processor, inputs)
    # Plot attention for each colour through layers
    # plt.figure(figsize=(10, 6))
    # for colour, bbox_attn in bbox_attn_per_layer.items():
    #     plt.plot(layers, bbox_attn, label=f"{colour.capitalize()} Bbox", marker='o')
    # plt.plot(baseline_attn, label="Baseline Attention", linestyle='--', color='grey')
    # plt.xlabel("Layer")
    # plt.ylabel("Attention to Bounding Box")
    # plt.title("Attention to Bounding Box Across Layers")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    item1_attention = bbox_attn_per_layer[colours[0]]
    item2_attention = bbox_attn_per_layer[colours[1]]

    return item1_attention, item2_attention, baseline_attn