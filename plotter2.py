import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from PIL import Image 
from pathlib import Path 
import json
import os 
import matplotlib.pyplot as plt

def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])

class Plotter:
    def __init__(self, image):
        self.image = image
        #self.load_model()
        img_path = Path(self.image)
        self.labels_path = img_path.parent.parent / "labels.json"
        W = 177
        with open(self.labels_path, 'r') as f:
            data = json.load(f)

        image_name = os.path.basename(img_path)
        shapes = []
        for item in data:
            if item['filename'] == image_name:
                for pair in item['shape_color_pairs']:
                    colour = pair['color']
                    shape = pair['shape']
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

                    position = next(pos for pos, col in item['spatial_arrangement'].items() if col == colour)
                    # store swapped ys because y1 is bigger than y2
                    shapes.append(self.Shape(colour, shape, [x1, y2, x2, y1], position))

            self.shapes = shapes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    class Shape:
        def __init__(self, colour, shape, bbox, position):
            self.colour = colour
            self.shape = shape
            self.bbox = bbox  # [x1, y1, x2, y2]
            self.position = position

    # def load_model(self):
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     global _loaded_models
    #     if '_loaded_models' not in globals():
    #         _loaded_models = {}

    #     model_id = "llava-hf/llava-1.5-7b-hf"
    #     if model_id not in _loaded_models:
    #         print("Loading model and processor...")
    #         processor = AutoProcessor.from_pretrained(model_id)
    #         model = LlavaForConditionalGeneration.from_pretrained(
    #             model_id, torch_dtype=torch.float16).to(device)
    #         _loaded_models[model_id] = (processor, model)
    #     else:
    #         print("Reusing cached model and processor.")
    #         processor, model = _loaded_models[model_id]

    #     self.model = model
    #     self.processor = processor

    def set_model(self, model, processor):
        self.model = model.to(self.device)
        self.processor = processor
        model.config.output_attentions = True
        if hasattr(self.model, "text_model"):
            self.model.text_model.config.output_attentions = True
        if hasattr(self.model, "vision_model"):
            self.model.vision_model.config.output_attentions = True


    def get_outputs(self, prompt):
        image = Image.open(self.image).convert("RGB")
        self.prompt = prompt


        inputs = self.processor(
            images=image,
            text=f"<IMG_CONTEXT>\n {self.prompt}",
            # truncation="only_second",
            padding=True,
            return_tensors='pt'
        )
        self.inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = self.inputs["input_ids"].to(self.device)
        # attention_mask = self.inputs["attention_mask"].to(self.device)
        # pixel_values = self.inputs["pixel_values"].to(self.device, dtype=torch.bfloat16)
        # self.inputs["pixel_values"] = pixel_values
        # with torch.no_grad():
        #     self.outputs = self.model(
        #         pixel_values=pixel_values,
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         output_attentions=True,
        #         attn_implementation="eager",
        #         return_dict=True
        #     )

        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "url": self.image},
        #             {"type": "text", "text": self.prompt}
        #         ]
        #     },
        # ]
        # self.inputs = self.processor.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     tokenize=True,
        #     return_dict=True,
        #     return_tensors="pt",
        # ).to(self.model.device)
        
        self.inputs["pixel_values"] = self.inputs["pixel_values"].to(self.device, dtype=torch.float16)

        self.outputs = self.model.generate(**self.inputs, max_new_tokens=1, do_sample=False, output_attentions=True, return_dict_in_generate=True)

        
        return self.outputs
    
    def print_output(self):
        input_len = self.inputs["input_ids"].shape[1]
        generated_text = self.processor.decode(self.outputs.sequences[0, input_len :], skip_special_tokens=True)
        # print all generated tokens
        return generated_text

    def get_matrix_all_layers(self):
        num_layers = len(self.outputs["attentions"][0])  # Get number of layers
        #print("Number of layers:", num_layers)
        aggregated_attn_all_layers_heads = []
        for l in range(num_layers):
            layer_attns = self.outputs["attentions"][0][l].squeeze(0)
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
        for out in self.outputs["attentions"]:
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
        
        # matrix should be input+output token length 
        # print token length
        output_token_len = len(self.outputs.sequences[0]) - len(self.inputs["input_ids"][0])
        input_token_len = len(self.inputs["input_ids"][0])
        total_token_len = input_token_len + output_token_len
        # print("Input token length:", input_token_len)
        # print("Output token length:", output_token_len)
        # print("Total token length:", total_token_len)
        # create final attention matrix with padded zeros
        return heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_attn_all_layers_heads) 
        + list(output_attentions))
    
    def get_matrix_for_layer(self, layer_idx):
        layer_attns = self.outputs["attentions"][0][layer_idx].squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        # Zero out attention to first <bos> token for non-first positions
        cur[1:, 0] = 0.
        # Normalize attention weights
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        input_attentions = cur


        # get attention from output tokens to rest of tokens
        output_attentions = []
        for out in self.outputs["attentions"]:   
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

        # matrix should be input+output token length 
        # print token length
        output_token_len = len(self.outputs.sequences[0]) - len(self.inputs["input_ids"][0])
        input_token_len = len(self.inputs["input_ids"][0])
        total_token_len = input_token_len + output_token_len
        # print("Input token length:", input_token_len)
        # print("Output token length:", output_token_len)
        # print("Total token length:", total_token_len)
        # print input tokens
        tokens = [self.processor.tokenizer.decode(i) for i in self.outputs.sequences[0]]
        #print("Tokens:", tokens)
        return heterogenous_stack(
            [torch.tensor([1])]  # Start with a dummy token
            + list(input_attentions)  # Add input attention
            + output_attentions  # Add output attentions
        )
    
    def get_image_attention_matrix(self, layer_idx):
        if layer_idx is None:
            matrix = self.get_matrix_all_layers()
        else:
            matrix = self.get_matrix_for_layer(layer_idx)


        print("full matrix element:")
        print(matrix[53][51])
        #print("Attention matrix shape:", matrix.shape)
        # plot matrix
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(matrix.cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Matrix (Layer {layer_idx if layer_idx is not None else "All Layers"})')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.savefig('attention_matrix.png')
        input_token_len = len(self.inputs["input_ids"][0])
        #print("Input token length:", input_token_len)
        output_token_len = len(self.outputs.sequences[0]) - input_token_len
        #print("Output token length:", output_token_len)
        tokens = [self.processor.tokenizer.decode(i) for i in self.outputs.sequences[0]]
        # print("Tokens:", tokens)
        # print("length tokens:", len(tokens))
        # if moel is llava image tok is <image> or if qwen image_tok is <|image_pad|>
        if "llava" in self.model.config._name_or_path:
            image_tok = '<image>'
        else:
            image_tok = '<IMG_CONTEXT>' # for internvl
        # get attention from output to image tokens
        output_indices = list(range(len(tokens) - output_token_len, len(tokens)))
        image_indices = [i for i, token in enumerate(tokens) if token == image_tok]
        # print("len(image_indices): ", len(image_indices))
        # print("output indices:", output_indices)
        # print("image indices:", image_indices)
        attn_out_to_image = matrix[output_indices][:,image_indices]

        # average over all output tokens
        attn_out_to_image = attn_out_to_image.mean(dim=0)
        # resize to image grid
        grid_size = int(len(image_indices) ** 0.5)
        
        return attn_out_to_image.reshape(grid_size, grid_size).cpu().numpy()

    def plot_image_attention(self, save_path, layer_idx=None):
        import matplotlib.pyplot as plt
        import numpy as np

        attn_matrix = self.get_image_attention_matrix(layer_idx)
        # print("Attention to image matrix shape:", attn_matrix.shape)
        #print("min and max of attn matrix:", attn_matrix.min(), attn_matrix.max())
        # # reset plot
        plt.clf()
        gmin = 0
        gmax = 0.5
        plt.imshow(attn_matrix, cmap='viridis')
        plt.colorbar()
        plt.title(f'Image Attention Map (Layer {layer_idx if layer_idx is not None else "All Layers"})')
        plt.axis('off')
        plt.savefig(save_path)

    def plot_image_attention_all_layers(self, save_path_prefix):
        num_layers = len(self.outputs["attentions"][0])  # Get number of layers
        for layer_idx in range(num_layers):
            save_path = f"{save_path_prefix}_layer_{layer_idx}.png"
            self.plot_image_attention(save_path, layer_idx=layer_idx)

    def plot_bbox_on_attention_map(self, save_path, layer_idx=None):
        import matplotlib.patches as patches
        import numpy as np
        import torch.nn.functional as F

        fig = plt.figure()
        attn_matrix = self.get_image_attention_matrix(layer_idx)
        
        attn_mask = np.zeros_like(attn_matrix)
        for shape in self.shapes:
            x1, y1, x2, y2 = shape.bbox
            # scale bounding box to attention heatmap size
            scale = attn_matrix.shape[0] / 177
            bbox_heatmap = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
            x1, y1, x2, y2 = bbox_heatmap

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # set attn_mask values within bbox to 1
            attn_mask[y1:y2+1, x1:x2+1] = 1.0

        img = Image.open(self.image).convert("RGB")
        img_numpy = np.array(img) / 255.0
        # print("Image shape:", img_numpy.shape)
        # print("Attention mask shape:", attn_mask.shape)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_numpy)
        attn_mask_resized = F.interpolate(
            torch.tensor(attn_mask).unsqueeze(0).unsqueeze(0),
            size=img.size[::-1],  # PIL uses (W, H), PyTorch uses (H, W)
            mode='bicubic',
            align_corners=False
        ).squeeze().numpy()
        ax.imshow(attn_mask_resized, cmap='viridis', alpha=0.7  )
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def get_bbox_attention_for_layer(self, shape, layer_idx):

        attn = self.get_image_attention_matrix(layer_idx)
        x1, y1, x2, y2 = shape.bbox
        # scale bounding box to attention heatmap size
        scale = attn.shape[0] / 177
        bbox_heatmap = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        x1, y1, x2, y2 = bbox_heatmap

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # calculate number of tokens in the bounding box

        bbox_tokens = (x2 - x1 + 1) * (y2 - y1 + 1)
        #print("number tokens in bbox", bbox_tokens)
        # plot attention heatmap but only for values within the bounding box
        return attn[y1:y2+1, x1:x2+1].sum().item()/bbox_tokens
            
    def get_baseline_attn_for_layer(self, shapes, layer_idx, no_token=False):
        attn = self.get_image_attention_matrix(layer_idx)
        total_attn = attn.sum().item()
        bbox_attn = 0.0
        total_bbox_tokens = 0
        for shape in shapes:
            x1, y1, x2, y2 = shape.bbox
            # scale bounding box to attention heatmap size
            scale = attn.shape[0] / 177
            bbox_heatmap = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
            x1, y1, x2, y2 = bbox_heatmap
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox_tokens = (x2 - x1 + 1) * (y2 - y1 + 1)
            total_bbox_tokens += bbox_tokens
            bbox_attn += attn[y1:y2+1, x1:x2+1].sum().item()
        
        baseline_attn = (total_attn - bbox_attn) / (attn.size - total_bbox_tokens)

        if no_token:
            return total_attn - bbox_attn
        else:
            return baseline_attn

    def plot_attention_through_layers(self, save_path=None):
        num_layers = len(self.outputs["attentions"][0])  # Get number of layers
        # read json 
        bbox_attentions = {shape.colour: [] for shape in self.shapes}
        for shape in self.shapes:
            bbox_attentions[shape.colour] = [self.get_bbox_attention_for_layer(shape, i) for i in range(num_layers)]

        baseline_attn = [self.get_baseline_attn_for_layer(self.shapes, i) for i in range(num_layers)]

        if save_path is not None:
            num_layers = len(self.outputs["attentions"][0])  # Get number of layers
            layers = list(range(num_layers))
            # Plot attention for each colour through layers
            plt.figure(figsize=(10, 6))
            for colour, bbox_attn in bbox_attentions.items():
                plt.plot(layers, bbox_attn, label=f"{colour.capitalize()} Bbox", marker='o', color=colour if colour != 'yellow' else 'gold')
            plt.plot(baseline_attn, label="Baseline Attention", linestyle='--', color='grey')
            plt.xlabel("Layer")
            plt.ylabel("Attention to Bounding Box")
            plt.title("Attention to Bounding Box Across Layers")
            plt.ylim(0, 0.007)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)

        return bbox_attentions, baseline_attn
    
    def plot_total_attention_to_image(self, save_path):
        num_layers = len(self.outputs["attentions"][0])  # Get number of layers
        total_attentions = []
        for layer_idx in range(num_layers):
            attn_matrix = self.get_image_attention_matrix(layer_idx)
            total_attentions.append(attn_matrix.sum().item())
        
        # Plot total attention to image through layers
        plt.figure(figsize=(10, 6))
        plt.plot(list(range(num_layers)), total_attentions, label="Total Attention to Image", marker='o', color='blue')
        plt.xlabel("Layer")
        plt.ylabel("Total Attention to Image")
        plt.title("Total Attention to Image Across Layers")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def get_left_shapes(self):
        return [shape for shape in self.shapes if 'left' in shape.position]
    
    def get_right_shapes(self):
        return [shape for shape in self.shapes if 'right' in shape.position]

    def get_shape_by_position(self, position):
        for shape in self.shapes:
            if shape.position == position:
                return shape
        return None