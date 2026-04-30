import os
from functions import load_model
from plotter import Plotter
import os
import numpy as np
import gc
import torch
import multiprocessing as mp
import argparse
from multiprocessing import Pool

def gen_2_data(model_name, image_dir, save_dir):

    processor, model = load_model(model_name)
    for neg in [False, True]:
        mentioned_attn = []
        non_mentioned_attn = []
        for image_path in os.listdir(image_dir+"/images/")[0:10]:
            plotter = Plotter(f"{image_dir}/images/{image_path}")
            left_colour = plotter.get_left_shapes()[0].colour
            right_colour = plotter.get_right_shapes()[0].colour

            # get attention for both left and right being mentioned and average
            for pos in ["left", "right"]:

                    if pos == "left":
                        mentioned_colour = left_colour
                        unmentioned_colour = right_colour
                    else:
                        mentioned_colour = right_colour
                        unmentioned_colour = left_colour

                    neg_val = "not " if neg else ""
                    article = "a " if image_dir == "whatsup" else ""
                    plotter.set_model(model, processor)
                    plotter.get_outputs(f"The figure is {neg_val}{article}{mentioned_colour}")
                    print(f"The figure is {neg_val}{article}{mentioned_colour}")
                    
                    bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                    mentioned_attn.append(bbox_attentions[mentioned_colour])
                    non_mentioned_attn.append(bbox_attentions[unmentioned_colour])

        save_neg = "neg_" if neg else ""
        # save data as numpy arrays
        np.save(save_dir + f"/{model_name}_{image_dir}_{save_neg}mentioned_attn.npy", np.array(mentioned_attn))
        np.save(save_dir + f"/{model_name}_{image_dir}_{save_neg}non_mentioned_attn.npy", np.array(non_mentioned_attn))



    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Finished data for {model_name}")

def gen_4_data(model_name, image_dir, save_dir):
    
    processor, model = load_model(model_name)

    

    for neg in [False, True]:
        mentioned_attn = []
        non_mentioned_attn = []
        for image_path in os.listdir(image_dir+"/images/")[0:10]:
            plotter = Plotter(f"{image_dir}/images/{image_path}")

            for pos in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:

                    top_left_colour = plotter.get_shape_by_position('top_left').colour
                    top_right_colour = plotter.get_shape_by_position('top_right').colour
                    bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
                    bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour
                    plotter.set_model(model, processor)
                    
                    if pos == 'top_left' or pos == 'neg_top_left':
                        target_colour = top_left_colour
                    elif pos == 'top_right' or pos == 'neg_top_right':
                        target_colour = top_right_colour
                    elif pos == 'bottom_left' or pos == 'neg_bottom_left':
                        target_colour = bottom_left_colour
                    elif pos == 'bottom_right' or pos == 'neg_bottom_right':
                        target_colour = bottom_right_colour

                    neg_val = "not " if neg else ""
                    plotter.get_outputs(f"The figure is {neg_val}{target_colour}")
                    print(f"The figure is {neg_val}{target_colour}")

                    bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                    if pos == 'top_left':
                        mentioned_attn.append(bbox_attentions[top_left_colour])
                        non_mentioned_attn.append([bbox_attentions[top_right_colour], bbox_attentions[bottom_left_colour], bbox_attentions[bottom_right_colour]])
                    elif pos == 'top_right':
                        mentioned_attn.append(bbox_attentions[top_right_colour])
                        non_mentioned_attn.append([bbox_attentions[top_left_colour], bbox_attentions[bottom_left_colour], bbox_attentions[bottom_right_colour]])
                    elif pos == 'bottom_left':  
                        mentioned_attn.append(bbox_attentions[bottom_left_colour])
                        non_mentioned_attn.append([bbox_attentions[top_left_colour], bbox_attentions[top_right_colour], bbox_attentions[bottom_right_colour]])
                    elif pos == 'bottom_right':
                        mentioned_attn.append(bbox_attentions[bottom_right_colour])
                        non_mentioned_attn.append([bbox_attentions[top_left_colour], bbox_attentions[top_right_colour], bbox_attentions[bottom_left_colour]])

                
        save_neg = "neg_" if neg else ""
        # save data as numpy arrays
        np.save(save_dir + f"/{model_name}_{image_dir}_{save_neg}mentioned_attn.npy", np.array(mentioned_attn))
        np.save(save_dir + f"/{model_name}_{image_dir}_{save_neg}non_mentioned_attn.npy", np.array(non_mentioned_attn))
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Finished data for {model_name}")

if __name__ == "__main__":
  # use multiprocessing for grab data function
    model_names = ['llava', 'internvl', 'paligemma']
    save_dir = "data_saves"
    parser = argparse.ArgumentParser(description="Generate data save files for all models")
    
    # Add the argument for image directory
    # 'default' provides a fallback, 'help' explains it in --help
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default="binary", 
        help="Path to the folder containing images"
    )
    
    args = parser.parse_args()

    target_func = gen_4_data if args.image_dir == "multary" else gen_2_data

    mp.set_start_method('spawn', force=True)
    # Use a context manager to handle the pool
    with Pool(processes=mp.cpu_count()) as pool:
        # Prepare the arguments for each call
        items = [(model, args.image_dir, save_dir) for model in model_names]
        
        # starmap allows passing multiple arguments to the function
        pool.starmap(target_func, items)

