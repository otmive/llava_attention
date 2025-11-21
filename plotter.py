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
        self.model = model
        self.processor = processor

    def get_outputs(self, prompt):
        image = Image.open(self.image).convert("RGB")
        self.prompt = prompt

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process the image
        processed = self.processor(images=image, text="", return_tensors="pt")
        #print("Processed pixel_values shape:", processed["pixel_values"].shape)

        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
        #print(len(inputs))
        self.inputs = inputs

        self.outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False, output_attentions=True, return_dict_in_generate=True)
        
        return self.outputs
    
    def print_output(self):
        generated_text = self.processor.batch_decode(self.outputs.sequences, skip_special_tokens=True)[0]
        #print(generated_text)
        return generated_text

    def get_matrix_all_layers(self):
        num_layers = len(self.outputs["attentions"][0])  # Get number of layers

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
        print("Input token length:", input_token_len)
        print("Output token length:", output_token_len)
        print("Total token length:", total_token_len)
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
        print("Input token length:", input_token_len)
        print("Output token length:", output_token_len)
        print("Total token length:", total_token_len)
        # print input tokens
        tokens = [self.processor.tokenizer.decode(i) for i in self.outputs.sequences[0]]
        print("Tokens:", tokens)
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

        print("Attention matrix shape:", matrix.shape)

        input_token_len = len(self.inputs["input_ids"][0])
        output_token_len = len(self.outputs.sequences[0]) - input_token_len
        tokens = [self.processor.tokenizer.decode(i) for i in self.outputs.sequences[0]]
        # if moel is llava image tok is <image> or if qwen image_tok is <|image_pad|>
        if "Qwen" in self.model.config._name_or_path:
            image_tok = '<|image_pad|>'
        else:
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

    def plot_image_attention(self, save_path, layer_idx=None):
        import matplotlib.pyplot as plt
        import numpy as np

        attn_matrix = self.get_image_attention_matrix(layer_idx)
        print("Attention to image matrix shape:", attn_matrix.shape)

        # reset plot
        plt.clf()
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
        print("number tokens in bbox", bbox_tokens)
        # plot attention heatmap but only for values within the bounding box
        return attn[y1:y2+1, x1:x2+1].sum().item()/bbox_tokens
        
    def get_baseline_attn_for_layer(self, shapes, layer_idx):
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
        return baseline_attn

    def plot_attention_through_layers(self, save_path=None):
        # read json 
        bbox_attentions = {shape.colour: [] for shape in self.shapes}
        for shape in self.shapes:
            bbox_attentions[shape.colour] = [self.get_bbox_attention_for_layer(shape, i) for i in range(32)]

        baseline_attn = [self.get_baseline_attn_for_layer(self.shapes, i) for i in range(32)]

        if save_path is not None:
            layers = list(range(32))
            # Plot attention for each colour through layers
            plt.figure(figsize=(10, 6))
            for colour, bbox_attn in bbox_attentions.items():
                plt.plot(layers, bbox_attn, label=f"{colour.capitalize()} Bbox", marker='o', color=colour if colour != 'yellow' else 'gold')
            plt.plot(baseline_attn, label="Baseline Attention", linestyle='--', color='grey')
            plt.xlabel("Layer")
            plt.ylabel("Attention to Bounding Box")
            plt.title("Attention to Bounding Box Across Layers")
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)

        return bbox_attentions, baseline_attn
    
    def get_left_shapes(self):
        return [shape for shape in self.shapes if 'left' in shape.position]
    
    def get_right_shapes(self):
        return [shape for shape in self.shapes if 'right' in shape.position]

    def get_shape_by_position(self, position):
        for shape in self.shapes:
            if shape.position == position:
                return shape
        return None