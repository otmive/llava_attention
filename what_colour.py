from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def binary_question(position, colour):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global _loaded_models
    if '_loaded_models' not in globals():
        _loaded_models = {}

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


    # item1_attn = []
    # item2_attn = []
    # baseline_attentions = []
    # for i in range(1000):
    #     image_path = f"2d_dataset_fixed_positions_1000/images/image_{i:04d}.png"
    #     plotter = Plotter(image_path)
    #     left_colour = plotter.get_left_shapes()[0].colour
    #     right_colour = plotter.get_right_shapes()[0].colour
    #     plotter.set_model(model, processor)
    #     plotter.get_outputs(f"The figure is not {right_colour}")
    #     bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
    #     item1_attn.append(bbox_attentions[left_colour])
    #     item2_attn.append(bbox_attentions[right_colour])
    #     baseline_attentions.append(baseline_attn)

    # # plot average attention across items
    # avg_item1_attn = np.mean(np.array(item1_attn), axis=0)
    # avg_item2_attn = np.mean(np.array(item2_attn), axis=0)
    # avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)
    # layers = list(range(32))
    # plt.figure(figsize=(10, 6))
    # plt.plot(layers, avg_item1_attn, label='Left Shape Attention', color='blue')
    # plt.plot(layers, avg_item2_attn, label='Right Shape Attention', color='red')
    # plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    # plt.xlabel('Layer')
    # plt.ylabel('Attention')
    # plt.title('Average Attention Scores Through Layers')
    # plt.legend()
    # # fix y axis limit
    # plt.ylim(0, 0.005)
    # plt.savefig('right_item_negated_average_attention.png')

    # image_path = "4_shapes_same_dataset_1000/images/image_0000.png"

    # plotter = Plotter(image_path)
    # top_left_colour = plotter.get_shape_by_position('top_left').colour

    # plotter.set_model(model, processor)
    # plotter.get_outputs(f"The figure is {top_left_colour}")
    # plotter.plot_attention_through_layers(save_path='top_left_item_positive_attention.png')


    correct_count = 0
    for i in range(10):
        image_path = f"4_shapes_same_dataset_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
        top_left_colour = plotter.get_shape_by_position('top_left').colour
        top_right_colour = plotter.get_shape_by_position('top_right').colour
        bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
        bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour
        # if top_left_colour == 'red':

        if position == 'top_left':
            target_colour = top_left_colour
            pos = 'top left'
        elif position == 'top_right':
            pos = 'top right'
            target_colour = top_right_colour
        elif position == 'bottom_left':
            target_colour = bottom_left_colour
            pos = 'bottom left'
        elif position == 'bottom_right':
            target_colour = bottom_right_colour
            pos = 'bottom right'


        plotter.set_model(model, processor)
        print(f"Is the {pos} shape {colour}? Answer in one word")
        plotter.get_outputs(f"Is the {pos} shape {colour}? Answer in one word")
        output = plotter.print_output().lower()
        print(f"Is the {pos} shape {colour}? output:", output)
        print("image:", image_path)

        if target_colour == colour and 'yes' in output:
            correct_count += 1
        elif target_colour != colour and 'no' in output:
            correct_count += 1

    accuracy = correct_count / 10
    print(f"Accuracy for position {position} and colour {colour}: {accuracy*100:.2f}%")
    return accuracy



if __name__ == "__main__":
    # loop through all position, coluor combinations
    positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    colours = ['red', 'blue', 'green', 'yellow']
    acc_by_position = {pos: [] for pos in positions}
    acc_by_colour = {col: [] for col in colours}
    all_acc = 0
    for pos in positions:
        total_acc = 0
        for col in colours:
            acc = binary_question(pos, col) 
            total_acc += acc
            print(f"Accuracy for position {pos} and colour {col}: {acc*100:.2f}%")
            all_acc += acc
            acc_by_colour[col].append(acc)
            acc_by_position[pos].append(acc)
        avg_acc = total_acc / len(colours)
        print(f"Average accuracy for position {pos}: {avg_acc*100:.2f}%")

    for col in colours:
        avg_acc = np.mean(acc_by_colour[col])
        print(f"Average accuracy for colour {col}: {avg_acc*100:.2f}%")
    for pos in positions:
        avg_acc = np.mean(acc_by_position[pos])
        print(f"Average accuracy for position {pos}: {avg_acc*100:.2f}%")

    # overall average
    overall_avg_acc = all_acc / (len(positions) * len(colours))
    print(f"Overall average accuracy: {overall_avg_acc*100:.2f}%")

    # plot accuracy by position
    position_names = list(acc_by_position.keys())
    position_acc = [np.mean(acc_by_position[pos]) for pos in position_names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(position_names, position_acc, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Position')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy by Position')
    plt.ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,   # x position
            height - 0.05,              # y position (slightly below bar)
            f"{height:.2f}",                   # format to 2 decimal places
            ha='center', va='bottom'
        )
    plt.savefig('accuracy_by_position.png')

    # plot accuracy by colour
    colour_names = list(acc_by_colour.keys())
    colour_acc = [np.mean(acc_by_colour[col]) for col in colour_names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(colour_names, colour_acc, color=['red', 'blue', 'green', 'gold'])
    plt.xlabel('Colour')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy by Colour')
    plt.ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,   # x position
            height - 0.05,              # y position (slightly below bar)
            f"{height:.2f}",                   # format to 2 decimal places
            ha='center', va='bottom'
        )
    plt.savefig('accuracy_by_colour.png')

# from plotter import Plotter
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration

# def binary_question(position, colour):
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


#     correct_count = 0
#     for i in range(1000):
#         image_path = f"2d_dataset_fixed_positions_1000/images/image_{i:04d}.png"
#         plotter = Plotter(image_path)
#         left_colour = plotter.get_left_shapes()[0].colour
#         right_colour = plotter.get_right_shapes()[0].colour
#         # if top_left_colour == 'red':

#         if position == 'left':
#             target_colour = left_colour
#             pos = 'left'
#         elif position == 'right':
#             pos = 'right'
#             target_colour = right_colour


#         plotter.set_model(model, processor)
#         plotter.get_outputs(f"Is the {pos} shape {colour}? Answer in one word")
#         output = plotter.print_output().lower()
        
#         if target_colour == colour and 'yes' in output:
#             correct_count += 1
#         elif target_colour != colour and 'no' in output:
#             correct_count += 1

#     accuracy = correct_count / 1000
#     print(f"Accuracy for position {position} and colour {colour}: {accuracy*100:.2f}%")
#     return accuracy



# if __name__ == "__main__":
#     # loop through all position, coluor combinations
#     positions = ['left', 'right']
#     colours = ['red', 'blue', 'green', 'yellow']
#     acc_by_position = {pos: [] for pos in positions}
#     acc_by_colour = {col: [] for col in colours}
#     all_acc = 0
#     for pos in positions:
#         total_acc = 0
#         for col in colours:
#             acc = binary_question(pos, col) 
#             total_acc += acc
#             all_acc += acc
#             acc_by_colour[col].append(acc)
#             acc_by_position[pos].append(acc)
#         avg_acc = total_acc / len(colours)
#         print(f"Average accuracy for position {pos}: {avg_acc*100:.2f}%")

#     for col in colours:
#         avg_acc = np.mean(acc_by_colour[col])
#         print(f"Average accuracy for colour {col}: {avg_acc*100:.2f}%")
#     for pos in positions:
#         avg_acc = np.mean(acc_by_position[pos])
#         print(f"Average accuracy for position {pos}: {avg_acc*100:.2f}%")

#     # overall average
#     overall_avg_acc = all_acc / (len(positions) * len(colours))
#     print(f"Overall average accuracy: {overall_avg_acc*100:.2f}%")

#     # plot accuracy by position
#     position_names = list(acc_by_position.keys())
#     position_acc = [np.mean(acc_by_position[pos]) for pos in position_names]
#     plt.figure(figsize=(8, 5))
#     bars = plt.bar(position_names, position_acc, color=['blue', 'orange'])
#     plt.xlabel('Position')
#     plt.ylabel('Average Accuracy')
#     plt.title('Average Accuracy by Position')
#     plt.ylim(0, 1)
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(
#             bar.get_x() + bar.get_width()/2,   # x position
#             height - 0.05,              # y position (slightly below bar)
#             f"{height:.2f}",                   # format to 2 decimal places
#             ha='center', va='bottom'
#         )
#     plt.savefig('accuracy_by_position.png')

#     # plot accuracy by colour
#     colour_names = list(acc_by_colour.keys())
#     colour_acc = [np.mean(acc_by_colour[col]) for col in colour_names]
#     plt.figure(figsize=(8, 5))
#     bars = plt.bar(colour_names, colour_acc, color=['red', 'blue', 'green', 'gold'])
#     plt.xlabel('Colour')
#     plt.ylabel('Average Accuracy')
#     plt.title('Average Accuracy by Colour')
#     plt.ylim(0, 1)
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(
#             bar.get_x() + bar.get_width()/2,   # x position
#             height - 0.05,              # y position (slightly below bar)
#             f"{height:.2f}",                   # format to 2 decimal places
#             ha='center', va='bottom'
#         )
#     plt.savefig('accuracy_by_colour.png')

