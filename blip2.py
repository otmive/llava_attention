import math
import torch

import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, InternVLForConditionalGeneration


def preprocess_image(pil_image, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(pil_image)

main_image = Image.open('2d_dataset_fixed_positions/images/image_0000.png').convert('RGB')

image_tensor = preprocess_image(main_image, image_size=448)
image_pil = transforms.ToPILImage()(image_tensor)
image_pil.save('./image_preprocessed.png')

image = Image.open('./image_preprocessed.png').convert('RGB')


device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
print(f"Using device: {device}")
model_name = 'OpenGVLab/InternVL3_5-4B-HF'

image_processor = AutoProcessor.from_pretrained(model_name)
inputs = image_processor(
    images=image,
    text="<IMG_CONTEXT>\n The figure is blue",
    truncation="only_second",
    padding=True,
    return_tensors='pt'
)


inputs = {k: v.to(device) for k, v in inputs.items()}
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)

model = InternVLForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    # device_map="auto",
    attn_implementation="eager"
).to(device)

model.config.output_attentions = True
if hasattr(model, "text_model"):
    model.text_model.config.output_attentions = True
if hasattr(model, "vision_model"):
    model.vision_model.config.output_attentions = True


with torch.no_grad():
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        attn_implementation="eager",
        return_dict=True
    )

all_attentions = outputs.attentions

IMAGE_TOKEN_INDEX = 151671
image_token_indecies = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1]
print("len(image_token_indecies): ", len(image_token_indecies))
print("len(all_attentions): ", len(all_attentions))
start_idx = image_token_indecies[0]
end_idx = image_token_indecies[-1]

start_idx, end_idx

n_layer_to_collect = 1
n_layer_to_collect = min(n_layer_to_collect, len(all_attentions))

attn_maps_all_layers = []
for layer_idx, layer_attn in enumerate(all_attentions[:n_layer_to_collect]):
    # layer_attn shape: (batch_size, num_heads, seq_len, seq_len)
    # Average across heads
    print("layer attn shape")
    print(layer_attn.shape)
    layer_attn_avg_head = layer_attn.mean(dim=1)
    print("layer attn avg head shape")
    print(layer_attn_avg_head.shape)
    # Extract attention from text tokens to vision tokens
    # Text tokens are after the image tokens (end_idx+1:)
    # Vision tokens are from start_idx to end_idx+1
    layer_attn_text_to_vis = layer_attn_avg_head[:, end_idx+1:, start_idx:end_idx+1]
    
    # Average over all text query positions
    layer_attn_vis_avg = layer_attn_text_to_vis.mean(dim=1)
    
    attn_maps_all_layers.append(layer_attn_vis_avg)

# Average across all collected layers
attention_map = torch.stack(attn_maps_all_layers, dim=0).mean(dim=0)

# Handle batch dimension
if attention_map.dim() > 1:
    attention_map = attention_map.squeeze(0)  # Remove batch dimension

print(f"Final attention map shape: {attention_map.shape}")

# Calculate grid size for spatial arrangement
num_patches = attention_map.shape[0]
grid_size = math.ceil(math.sqrt(num_patches))

print(f"Number of vision tokens: {num_patches}")
print(f"Grid size: {grid_size}x{grid_size}")

# Pad at the END, not the beginning
if grid_size * grid_size != num_patches:
    pad_size = grid_size * grid_size - num_patches
    print(f"Padding with {pad_size} zeros at the END")
    
    zero_pad = torch.zeros(pad_size, dtype=attention_map.dtype, device=attention_map.device)
    attention_map = torch.cat([attention_map, zero_pad], dim=0)

# Normalize attention map
attention_map = attention_map.view(grid_size, grid_size)
attention_map = attention_map / (attention_map.sum(dim=-1, keepdim=True) + 1e-8)


def create_attention_visualization(original_image, attention_map, output_path):
    """Create and save attention map visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # # Attention map
    # attention_resized = F.interpolate(
    #     attention_map.unsqueeze(0).unsqueeze(0).to(torch.float32),
    #     size=original_image.size[::-1],
    #     mode='bilinear',
    #     align_corners=False
    # ).squeeze().cpu().numpy()

    print("min and max of attention map: ", attention_map.min().item(), attention_map.max().item())
    im1 = axes[1].imshow(attention_map.float().cpu().numpy(), alpha=0.8)
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    # Overlay
    # axes[2].imshow(original_image)
    # axes[2].imshow(attention_resized, alpha=0.5)
    # axes[2].set_title('Attention Overlay')
    # axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Attention visualization saved to: {output_path}")

create_attention_visualization(main_image, attention_map, f"attention_visualization-{model_name.split('/')[1]}.png")