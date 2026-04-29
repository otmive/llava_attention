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
        self.dataset_type = "2d"
        img_path = Path(self.image)
        self.labels_path = img_path.parent.parent / "labels.json"
        # check labels path exists 
        if not self.labels_path.exists():
            self.labels_path = img_path.parent.parent / "whatsup_labels.json"
            self.dataset_type = "real"
        max_size = 448
        self.img = Image.open(self.image).convert("RGB")
        orig_W, orig_H = self.img.size
        if self.img.size[0] > max_size or self.img.size[1] > max_size:
            self.img = self.img.resize((max_size, max_size))
        W, H = self.img.size
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
                    
                    if self.dataset_type == "2d":
                    # initial size of 2d images when generated
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
                    elif self.dataset_type == "real":
                        scale_x = W / orig_W
                        scale_y = H / orig_H

                        x1, y1, x2, y2 = bbox
                        x1 *= scale_x
                        x2 *= scale_x
                        y1 *= scale_y
                        y2 *= scale_y

                    position = next(pos for pos, col in item['spatial_arrangement'].items() if col == colour)
                    # store swapped ys because y1 is bigger than y2
                    if self.dataset_type == "2d":
                        bbox = [x1, y2, x2, y1]
                    else:
                        bbox = [x1, y1, x2, y2]
                    shapes.append(self.Shape(colour, shape, bbox, position))

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
        #print("in get outputs")
        image = Image.open(self.image).convert("RGB")
        image = self.img
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

        # check if model is internvl or llava
        #print("checking which model")
        if "InternVL" in self.model.config._name_or_path:
            #print("using internvl processing")
                
            inputs = self.processor(
                images=image,
                text=f"<IMG_CONTEXT>\nUser: {self.prompt}\nAssistant:",
                # truncation="only_second",
                padding=True,
                return_tensors='pt'
            ).to(self.device, torch.float16)
            self.inputs = {k: v.to(self.device) for k, v in inputs.items()}
            #input_ids = self.inputs["input_ids"].to(self.device)
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    print(k, v.device)
            self.inputs["pixel_values"] = self.inputs["pixel_values"].to(self.device)

        elif "llava" in self.model.config._name_or_path:

            #print("using llava processing")

            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # Process the image
            processed = self.processor(images=image, text="", return_tensors="pt")
            #print("Processed pixel_values shape:", processed["pixel_values"].shape)

            inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.device, torch.float16)
            #print(len(inputs))
            self.inputs = inputs

        elif "paligemma" in self.model.config._name_or_path:
            #print("using paligemma processing")


            self.inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(self.device)
        

        
        # print model device
        #print("Model device:", next(self.model.parameters()).device)

        self.outputs = self.model.generate(**self.inputs, max_new_tokens=1, do_sample=False, output_attentions=True, return_dict_in_generate=True)
        
        return self.outputs
    
    def print_output(self):
        ## check model type
        if "InternVL" in self.model.config._name_or_path:
            # Get just the last 10 tokens where the actual text is
            # tail_tokens = self.outputs.sequences[0][-10:]
            
            # print("--- Debugging Tokens ---")
            # for token_id in tail_tokens:
            #     # convert_ids_to_tokens shows the literal string representation (e.g., '<|extra_0|>')
            #     token_string = self.processor.tokenizer.convert_ids_to_tokens(token_id.item())
            #     print(f"ID: {token_id.item()} -> String: '{token_string}'")
            
            # This is what you actually want:
            input_len = self.inputs['input_ids'].shape[1]
            generated_text = self.processor.tokenizer.decode(self.outputs.sequences[0][input_len:])
        else:
            input_len = self.inputs['input_ids'].shape[1]
            generated_text = self.processor.batch_decode(self.outputs.sequences[:, input_len:], skip_special_tokens=True)[0]

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
        # print("Input token length:", input_token_len)
        # print("Output token length:", output_token_len)
        # print("Total token length:", total_token_len)
        # create final attention matrix with padded zeros
        return heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_attn_all_layers_heads) 
        + list(output_attentions))
    
    def get_matrix_for_layer_and_head(self, layer_idx, head_idx):
        layer_attns = self.outputs["attentions"][0][layer_idx].squeeze(0)
        attns_per_head = layer_attns[head_idx]
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
            attns_per_head = layer_attns[head_idx]
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
    
    def get_image_attention_matrix(self, layer_idx, head_idx=None):
        if layer_idx is None and head_idx is None:
            matrix = self.get_matrix_all_layers()
        elif layer_idx is not None and head_idx is None:
            matrix = self.get_matrix_for_layer(layer_idx)
        else:
            matrix = self.get_matrix_for_layer_and_head(layer_idx, head_idx)

        #print("Attention matrix shape:", matrix.shape)

        input_token_len = len(self.inputs["input_ids"][0])
        output_token_len = len(self.outputs.sequences[0]) - input_token_len
        tokens = [self.processor.tokenizer.decode(i) for i in self.outputs.sequences[0]]
        # if moel is llava image tok is <image> or if qwen image_tok is <|image_pad|>
        if "llava" in self.model.config._name_or_path:
            image_tok = '<image>'
        elif "InternVL" in self.model.config._name_or_path:
            image_tok = '<IMG_CONTEXT>' # for internvl
        elif "paligemma" in self.model.config._name_or_path:
            image_tok = '<image>'
        # get attention from output to image tokens
        output_indices = list(range(len(tokens) - output_token_len, len(tokens)))
        image_indices = [i for i, token in enumerate(tokens) if token == image_tok]
        #print("No. of image tokens:", len(image_indices))
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
        #print("Attention to image matrix shape:", attn_matrix.shape)

        # reset plot
        plt.clf()
        plt.imshow(attn_matrix, cmap='viridis')
        plt.colorbar()
        plt.title(f'Image Attention Map (Layer {layer_idx if layer_idx is not None else "All Layers"})')
        plt.axis('off')
        plt.savefig(save_path)

    def plot_attention_for_head(self, layer_idx, head_idx, save_path=None):
        import matplotlib.pyplot as plt
        import numpy as np

        attn_matrix = self.get_matrix_for_layer_and_head(layer_idx, head_idx)
        #print("Attention to image matrix shape:", attn_matrix.shape)

        # get attention from output tokens to image tokens
        input_token_len = len(self.inputs["input_ids"][0])
        output_token_len = len(self.outputs.sequences[0]) - input_token_len
        tokens = [self.processor.tokenizer.decode(i) for i in self.outputs.sequences[0]]
        # if moel is llava image tok is <image> or if qwen image_tok is <|image_pad|>
        if "llava" in self.model.config._name_or_path:
            image_tok = '<image>'
        elif "InternVL" in self.model.config._name_or_path:
            image_tok = '<IMG_CONTEXT>' # for internvl
        elif "paligemma" in self.model.config._name_or_path:
            image_tok = '<image>'
        output_indices = list(range(len(tokens) - output_token_len, len(tokens)))
        image_indices = [i for i, token in enumerate(tokens) if token == image_tok]
        #print("No. of image tokens:", len(image_indices))
        attn_out_to_image = attn_matrix[output_indices][:,image_indices]

        # average over all output tokens
        attn_out_to_image = attn_out_to_image.mean(dim=0)
        # resize to image grid
        grid_size = int(len(image_indices) ** 0.5)
        attn_image = attn_out_to_image.reshape(grid_size, grid_size).cpu().numpy()

        # reset plot
        if save_path is not None:
            plt.clf()
            plt.imshow(attn_image, cmap='viridis')
            plt.colorbar()
            plt.title(f'Image Attention Map (Layer {layer_idx}, Head {head_idx})')
            plt.axis('off')
            plt.savefig(save_path)


    def plot_bbox_on_attention_map(self, save_path, layer_idx=None):
        import matplotlib.patches as patches
        import numpy as np
        import torch.nn.functional as F

        fig = plt.figure()
        attn_matrix = self.get_image_attention_matrix(layer_idx)
        
        img = Image.open(self.image).convert("RGB")

        attn_mask = np.zeros_like(attn_matrix)
        for shape in self.shapes:
            x1, y1, x2, y2 = shape.bbox
            # scale bounding box to attention heatmap size
            #scale = attn_matrix.shape[0] / 177

            # set scale based on image size 1280x960
            scale_x = attn_matrix.shape[1] / img.size[0]
            scale_y = attn_matrix.shape[0] / img.size[1]

            bbox_heatmap = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            x1, y1, x2, y2 = bbox_heatmap

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # set attn_mask values within bbox to 1
            attn_mask[y1:y2+1, x1:x2+1] = 1.0

        
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

    def plot_image_attention_all_layers(self, save_path_prefix):
        num_layers = len(self.outputs["attentions"][0])  # Get number of layers
        for layer_idx in range(num_layers):
            save_path = f"{save_path_prefix}_layer_{layer_idx}.png"
            self.plot_image_attention(save_path, layer_idx=layer_idx)

    def plot_bbox_on_image(self, save_path):
        import matplotlib.patches as patches
        import numpy as np

        fig = plt.figure()
        img_numpy = np.array(self.img) / 255.0
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_numpy)

        for shape in self.shapes:
            x1, y1, x2, y2 = shape.bbox
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 10, shape.shape, color='red', fontsize=12, weight='bold')

        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)


    def plot_bbox_on_attention_map(self, save_path, layer_idx=None):
        import matplotlib.patches as patches
        import numpy as np
        import torch.nn.functional as F

        fig = plt.figure()
        attn_matrix = self.get_image_attention_matrix(layer_idx)
        
        #img = Image.open(self.image).convert("RGB")

        attn_mask = np.zeros_like(attn_matrix)
        for shape in self.shapes:
            x1, y1, x2, y2 = shape.bbox
            # scale bounding box to attention heatmap size
            #scale = attn_matrix.shape[0] / 177

            # set scale based on image size 1280x960
            scale_x = attn_matrix.shape[0] / self.img.size[0]
            scale_y = attn_matrix.shape[1] / self.img.size[1]


            bbox_heatmap = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            x1, y1, x2, y2 = bbox_heatmap

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # print("original bbox:", shape.bbox)
            # print("scaled bbox:", [x1, y1, x2, y2])
            # print("attention map shape:", attn_matrix.shape)
            # print("image shape:", self.img.size)

            # set attn_mask values within bbox to 1
            attn_mask[y1:y2+1, x1:x2+1] = 1.0

            # print("attention mask after bbox:", attn_mask)

        
        img_numpy = np.array(self.img) / 255.0
        # print("Image shape:", img_numpy.shape)
        # print("Attention mask shape:", attn_mask.shape)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_numpy)
        attn_mask_resized = F.interpolate(
            torch.tensor(attn_mask).unsqueeze(0).unsqueeze(0),
            size=self.img.size[::-1],  # PIL uses (W, H), PyTorch uses (H, W)
            mode='bicubic',
            align_corners=False
        ).squeeze().numpy()
        #print("Resized attention mask shape:", attn_mask_resized.shape)
        ax.imshow(attn_mask_resized, cmap='viridis', alpha=0.7  )
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def get_bbox_attention_for_layer_total(self, shape, layer_idx):

        attn = self.get_image_attention_matrix(layer_idx)
        # print(f"attention for layer {layer_idx}:", attn[0:5,0:5])
        x1, y1, x2, y2 = shape.bbox
        # scale bounding box to attention heatmap size
        scale_x = attn.shape[1] / self.img.size[0]
        scale_y = attn.shape[0] / self.img.size[1]

        bbox_heatmap = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        x1, y1, x2, y2 = bbox_heatmap

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # print("original bbox:", shape.bbox)
        # print("scaled bbox:", [x1, y1, x2, y2])
        # print("attention map shape:", attn.shape)
        # print("image shape:", self.img.size)
        # calculate number of tokens in the bounding box

        bbox_tokens = (x2 - x1 + 1) * (y2 - y1 + 1)
        #print("number tokens in bbox", bbox_tokens)
        # plot attention heatmap but only for values within the bounding box
        return attn[y1:y2+1, x1:x2+1].sum().item()
    
    def get_bbox_attention_for_layer(self, shape, layer_idx):

        attn = self.get_image_attention_matrix(layer_idx)
        # print(f"attention for layer {layer_idx}:", attn[0:5,0:5])
        x1, y1, x2, y2 = shape.bbox
        # scale bounding box to attention heatmap size
        scale_x = attn.shape[1] / self.img.size[0]
        scale_y = attn.shape[0] / self.img.size[1]

        bbox_heatmap = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        x1, y1, x2, y2 = bbox_heatmap

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # print("original bbox:", shape.bbox)
        # print("scaled bbox:", [x1, y1, x2, y2])
        # print("attention map shape:", attn.shape)
        # print("image shape:", self.img.size)
        # calculate number of tokens in the bounding box

        bbox_tokens = (x2 - x1 + 1) * (y2 - y1 + 1)
        # plot attention heatmap but only for values within the bounding box
        return attn[y1:y2+1, x1:x2+1].sum().item()/bbox_tokens
        
    def get_baseline_attn_for_layer(self, shapes, layer_idx):
        attn = self.get_image_attention_matrix(layer_idx)
        total_attn = attn.sum().item()
        #print("Total attention:", total_attn)
        bbox_attn = 0.0
        total_bbox_tokens = 0
        for shape in shapes:
            x1, y1, x2, y2 = shape.bbox
            # scale bounding box to attention heatmap size
            scale_x = attn.shape[1] / self.img.size[0]
            scale_y = attn.shape[0] / self.img.size[1]

            bbox_heatmap = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            x1, y1, x2, y2 = bbox_heatmap
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox_tokens = (x2 - x1 + 1) * (y2 - y1 + 1)
            total_bbox_tokens += bbox_tokens
            bbox_attn += attn[y1:y2+1, x1:x2+1].sum().item()
        baseline_attn = (total_attn - bbox_attn) / (attn.size - total_bbox_tokens)
        return baseline_attn
    
    def get_baseline_attn_for_layer_total(self, shapes, layer_idx):
        attn = self.get_image_attention_matrix(layer_idx)
        total_attn = attn.sum().item()
        #print("Total attention:", total_attn)
        bbox_attn = 0.0
        total_bbox_tokens = 0
        for shape in shapes:
            x1, y1, x2, y2 = shape.bbox
            # scale bounding box to attention heatmap size
            scale_x = attn.shape[1] / self.img.size[0]
            scale_y = attn.shape[0] / self.img.size[1]

            bbox_heatmap = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            x1, y1, x2, y2 = bbox_heatmap
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox_tokens = (x2 - x1 + 1) * (y2 - y1 + 1)
            total_bbox_tokens += bbox_tokens
            bbox_attn += attn[y1:y2+1, x1:x2+1].sum().item()
        
        baseline_attn = (total_attn - bbox_attn) 
        return baseline_attn

    def plot_attention_through_layers_total(self, save_path=None, ylimit=None):
        num_layers = len(self.outputs["attentions"][0])
        # read json 
        bbox_attentions = {shape.colour: [] for shape in self.shapes}
        for shape in self.shapes:
            #print(f"Calculating attention for shape: {shape.shape} ({shape.colour})")
            bbox_attentions[shape.colour] = [self.get_bbox_attention_for_layer_total(shape, i) for i in range(num_layers)]

        baseline_attn = [self.get_baseline_attn_for_layer_total(self.shapes, i) for i in range(num_layers)]

        if save_path is not None:
            num_layers = len(self.outputs["attentions"][0])
            layers = list(range(num_layers))
            # Plot attention for each colour through layers
            
            plt.figure(figsize=(10, 6))
            for colour, bbox_attn in bbox_attentions.items():
                if colour in ["red", "blue", "green", "yellow"]:
                    colour = colour if colour != 'yellow' else 'gold'
                
                plt.plot(layers, bbox_attn, label=f"{colour.capitalize()} Shape", marker='o', color=colour if colour in ["red", "blue", "green", "gold"] else None)
            plt.plot(baseline_attn, label="Baseline Attention", linestyle='--', color='grey')
            plt.xlabel("Layer")
            plt.ylabel("Attention to Bounding Box")
            plt.title("Attention to Bounding Box Across Layers")
            if ylimit is not None:
                plt.ylim(ylimit)    
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return bbox_attentions, baseline_attn
    
    def plot_attention_through_layers(self, save_path=None, ylimit=None):
        num_layers = len(self.outputs["attentions"][0])
        # read json 
        bbox_attentions = {shape.colour: [] for shape in self.shapes}
        for shape in self.shapes:
            #print(f"Calculating attention for shape: {shape.shape} ({shape.colour})")
            bbox_attentions[shape.colour] = [self.get_bbox_attention_for_layer(shape, i) for i in range(num_layers)]

        baseline_attn = [self.get_baseline_attn_for_layer(self.shapes, i) for i in range(num_layers)]

        if save_path is not None:
            num_layers = len(self.outputs["attentions"][0])
            layers = list(range(num_layers))
            # Plot attention for each colour through layers
            
            plt.figure(figsize=(10, 6))
            for colour, bbox_attn in bbox_attentions.items():
                if colour in ["red", "blue", "green", "yellow"]:
                    colour = colour if colour != 'yellow' else 'gold'
                
                plt.plot(layers, bbox_attn, label=f"{colour.capitalize()} Shape", marker='o', color=colour if colour in ["red", "blue", "green", "gold"] else None)
            plt.plot(baseline_attn, label="Baseline Attention", linestyle='--', color='grey')
            plt.xlabel("Layer")
            plt.ylabel("Attention to Bounding Box")
            plt.title("Attention to Bounding Box Across Layers")
            if ylimit is not None:
                plt.ylim(ylimit)    
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return bbox_attentions, baseline_attn
    
    def get_total_image_attention(self):
        attn = self.get_image_attention_matrix(None)
        return attn.sum().item()
    
    def get_left_shapes(self):
        return [shape for shape in self.shapes if 'left' in shape.position]
    
    def get_right_shapes(self):
        return [shape for shape in self.shapes if 'right' in shape.position]

    def get_shape_by_position(self, position):
        for shape in self.shapes:
            if shape.position == position:
                return shape
        return None
    
    # def get_outputs_downstream(self, prompt):
    #     #print("in get outputs")
    #     image = Image.open(self.image).convert("RGB")
    #     image = self.img
    #     self.prompt = prompt

    #     conversation = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": self.prompt},
    #                 {"type": "image"},
    #             ],
    #         },
    #     ]

    #     # check if model is internvl or llava
    #     #print("checking which model")
    #     if "InternVL" in self.model.config._name_or_path:
    #         #print("using internvl processing")
                
    #         inputs = self.processor(
    #             images=image,
    #             text=f"<IMG_CONTEXT>\nUser: {self.prompt}\nAssistant:",
    #             # truncation="only_second",
    #             padding=True,
    #             return_tensors='pt'
    #         ).to(self.device, torch.float16)
    #         self.inputs = {k: v.to(self.device) for k, v in inputs.items()}
    #         #input_ids = self.inputs["input_ids"].to(self.device)
    #         for k, v in inputs.items():
    #             if torch.is_tensor(v):
    #                 print(k, v.device)
    #         self.inputs["pixel_values"] = self.inputs["pixel_values"].to(self.device)

    #     elif "llava" in self.model.config._name_or_path:

    #         #print("using llava processing")

    #         prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
    #         # Process the image
    #         processed = self.processor(images=image, text="", return_tensors="pt")
    #         #print("Processed pixel_values shape:", processed["pixel_values"].shape)

    #         inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.device, torch.float16)
    #         #print(len(inputs))
    #         self.inputs = inputs

    #     elif "paligemma" in self.model.config._name_or_path:
    #         #print("using paligemma processing")


    #         self.inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(self.device)
        

        
    #     # print model device
    #     #print("Model device:", next(self.model.parameters()).device)

    #     self.outputs = self.model.generate(**self.inputs, max_new_tokens=10, do_sample=False, output_attentions=True, return_dict_in_generate=True)
        
    #     return self.outputs