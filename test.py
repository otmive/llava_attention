from plotter2 import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration


def four_position(pos, colour):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # global _loaded_models
    # if '_loaded_models' not in globals():
    #     _loaded_models = {}

    # model_id = "llava-hf/llava-1.5-7b-hf"
    # if model_id not in _loaded_models:
    #     print("Loading model and processor...")
    #     processor = AutoProcessor.from_pretrained(model_id)
    #     model = LlavaForConditionalGeneration.from_pretrained(
    #         model_id, torch_dtype=torch.float16).to(device)
    #     _loaded_models[model_id] = (processor, model)
    # else:
    #     print("Reusing cached model and processor.")
    #     processor, model = _loaded_models[model_id]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global _loaded_models
    if '_loaded_models' not in globals():
        _loaded_models = {}

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


    item1_attn = []
    item2_attn = []
    baseline_attentions = []
    for i in range(1000):
        image_path = f"2d_dataset_fixed_positions_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour
        plotter.set_model(model, processor)

        if pos == 'left' or pos == 'neg_left':
            target_colour = left_colour
        elif pos == 'right' or pos == 'neg_right':
            target_colour = right_colour

        if target_colour == colour:

            if pos == 'left':
                plotter.get_outputs(f"The figure is {left_colour}")
            elif pos == 'right':
                plotter.get_outputs(f"The figure is {right_colour}")
            elif pos == 'neg_left':
                plotter.get_outputs(f"The figure is not {left_colour}")
            elif pos == 'neg_right':
                plotter.get_outputs(f"The figure is not {right_colour}")

            print("outputs:", plotter.print_output())
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            item1_attn.append(bbox_attentions[left_colour])
            item2_attn.append(bbox_attentions[right_colour])
            baseline_attentions.append(baseline_attn)


    # plot average attention across items
    avg_item1_attn = np.mean(np.array(item1_attn), axis=0)
    avg_item2_attn = np.mean(np.array(item2_attn), axis=0)
    avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)
    std_item1_attn = np.std(np.array(item1_attn), axis=0)
    std_item2_attn = np.std(np.array(item2_attn), axis=0)
    layers = list(range(36))
    plt.figure(figsize=(10, 6))
    plt.plot(layers, avg_item1_attn, label='Left Shape Attention', color='blue')
    plt.plot(layers, avg_item2_attn, label='Right Shape Attention', color='red')
    plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    # plot confidence intervals
    #plt.fill_between(layers, avg_item1_attn - std_item1_attn, avg_item1_attn + std_item1_attn, color='blue', alpha=0.2)
    #plt.fill_between(layers, avg_item2_attn - std_item2_attn, avg_item2_attn + std_item2_attn, color='red', alpha=0.2)    
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Average Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, 0.04)
    plt.savefig(f'intern_colour_plots/{pos}_{colour}all.png')

    # image_path = "4_shapes_same_dataset_1000/images/image_0000.png"

    # plotter = Plotter(image_path)
    # top_left_colour = plotter.get_shape_by_position('top_left').colour

    # plotter.set_model(model, processor)
    # plotter.get_outputs(f"The figure is {top_left_colour}")
    # plotter.plot_attention_through_layers(save_path='top_left_item_positive_attention.png')


#     item1_attn = []
#     item2_attn = []
#     item3_attn = []
#     item4_attn = []
#     baseline_attentions = []
#     for i in range(1000):
#         image_path = f"4_shapes_same_dataset_1000/images/image_{i:04d}.png"
#         plotter = Plotter(image_path)
#         top_left_colour = plotter.get_shape_by_position('top_left').colour
#         top_right_colour = plotter.get_shape_by_position('top_right').colour
#         bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
#         bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour
#         # if top_left_colour == 'red':
#         #     target_colour = top_left_colour
#         if pos == 'bottom_left':
#             target_colour = bottom_left_colour
#         elif pos == 'bottom_right':
#             target_colour = bottom_right_colour
#         elif pos == 'top_left':
#             target_colour = top_left_colour
#         elif pos == 'top_right':
#             target_colour = top_right_colour

#         if target_colour == 'yellow':
#             print(f"Processing image {i}: TL={top_left_colour}, TR={top_right_colour}, BL={bottom_left_colour}, BR={bottom_right_colour}")
#             plotter.set_model(model, processor)
#             plotter.get_outputs(f"The figure is not {target_colour}")
#             bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
#             item1_attn.append(bbox_attentions[top_left_colour])
#             item2_attn.append(bbox_attentions[top_right_colour])
#             item3_attn.append(bbox_attentions[bottom_left_colour])
#             item4_attn.append(bbox_attentions[bottom_right_colour])
#             baseline_attentions.append(baseline_attn)
#         # clear figure
#         # plt.clf()
#         # plt.figure(figsize=(10, 6))
#         # for colour, bbox_attn in bbox_attentions.items():
#         #     plt.plot(list(range(32)), bbox_attn, label=f"{colour.capitalize()} Bbox", marker='o', color=colour if colour != 'yellow' else 'gold')
#         # plt.plot(baseline_attn, label="Baseline Attention", linestyle='--', color='grey')
#         # plt.xlabel("Layer")
#         # plt.ylabel("Attention to Bounding Box")
#         # plt.title(f'Attention Scores Through Layers for Image {i}')
#         # plt.legend()
#         # plt.ylim(0, 0.005)
#         # plt.savefig(f'4_shapes_image_{i:04d}_attention.png')

#     # plot average attention across items
#     avg_item1_attn = np.mean(np.array(item1_attn), axis=0)
#     avg_item2_attn = np.mean(np.array(item2_attn), axis=0)
#     avg_item3_attn = np.mean(np.array(item3_attn), axis=0)
#     avg_item4_attn = np.mean(np.array(item4_attn), axis=0)
#     avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0
#     )
#     layers = list(range(32))
#     plt.figure(figsize=(10, 6))
#     plt.plot(layers, avg_item1_attn, label='Top Left Shape Attention')
#     plt.plot(layers, avg_item2_attn, label='Top Right Shape Attention')
#     plt.plot(layers, avg_item3_attn, label='Bottom Left Shape Attention')
#     plt.plot(layers, avg_item4_attn, label='Bottom Right Shape Attention')
#     plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
#     plt.xlabel('Layer')
#     plt.ylabel('Attention')
#     plt.title('Average Attention Scores Through Layers')
#     plt.legend()
#     # fix y axis limit
#     plt.ylim(0, 0.007)
#     plt.savefig(f'Images/{pos}_yellow_negated_1000_average_attention.png')


    # image_path = "2d_dataset_fixed_positions_1000/images/image_0004.png"
    # plotter = Plotter(image_path)
    # left_colour = plotter.get_left_shapes()[0].colour
    # right_colour = plotter.get_right_shapes()[0].colour
    # plotter.set_model(model, processor)
    # plotter.get_outputs(f"The figure is not {left_colour}")
    # for i in range(32):
    #     plotter.plot_image_attention(save_path=f'image_0004/{i}.png', layer_idx=i)

    # plotter.plot_attention_through_layers(save_path=f'image_0004/attention_neg.png')

if __name__ == "__main__":
#     four_position('top_left')
#     four_position('top_right')
#     four_position('bottom_left')
#     four_position('bottom_right')
    #four_position('left', 'green')
    four_position('left', 'red')
    four_position('left', 'blue')
    four_position('left', 'yellow')
    four_position('right', 'green')
    four_position('right', 'red')
    four_position('right', 'blue')
    four_position('right', 'yellow')