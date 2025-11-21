from functions import plot_attention_through_layers
import json 
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(colour):
    first_item_attention = []
    second_item_attention = []
    base_attention = []
    count = 0
    for i in range(10):
        image_data = "2d_dataset_vertical_positions/labels.json"
        with open(image_data, 'r') as f:
            data = json.load(f)
        image_url = f"2d_dataset_vertical_positions/images/image_{i:04d}.png"
        image_name = os.path.basename(image_url)
        
        for item in data:
            if item['filename'] == image_name:
                colours = [pair['color'] for pair in item['shape_color_pairs']]


        print("Processing image:", image_url)
        item1, item2, base = plot_attention_through_layers(image_url, "right")
        first_item_attention.append(item1)
        second_item_attention.append(item2)
        print(base)
        base_attention.append(base)
        count+=1


    # Convert lists to numpy arrays for easier manipulation
    first_item_attention = np.array(first_item_attention)
    second_item_attention = np.array(second_item_attention)
    # np.save(f"attention_saves/left_all.npy", first_item_attention)
    # np.save(f"attention_saves/left_all_second.npy", second_item_attention)
    # np.save(f"attention_saves/left_all_base.npy", base_attention)
    base_attention = np.array(base_attention)
    print(base_attention.shape)
    # Plot average attention for each item across all images
    avg_first_item_attention = np.mean(first_item_attention, axis=0)
    avg_second_item_attention = np.mean(second_item_attention, axis=0)
    avg_base_attention = np.mean(base_attention, axis=0)
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 0.007)
    plt.plot(range(32), avg_first_item_attention, label="Average Top Item Attention", marker='o', color="blue" )
    plt.plot(range(32), avg_second_item_attention, label="Average Bottom Item Attention", marker='o', color = "orange")
    plt.plot(range(32), avg_base_attention, label="Average Baseline Attention", linestyle='--', color='grey')
    plt.xlabel("Layer")
    plt.ylabel("Average Attention to Bounding Box")
    plt.title(f"Average Attention All Images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"all_10_bottom.png")
    print("count", count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colour", type=str, default="green", help="Colour to filter images by (e.g., 'green')")
    args = parser.parse_args()
    main(args.colour)