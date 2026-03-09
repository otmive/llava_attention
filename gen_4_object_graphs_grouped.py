from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration, AutoModelForImageTextToText
import argparse
import multiprocessing as mp

def load_model(model_name):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global _loaded_models
    if '_loaded_models' not in globals():
        _loaded_models = {}

    if model_name == "internvl":
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
    
    elif model_name == "llava":
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

    elif model_name == "paligemma":

        model_id = "google/paligemma2-3b-mix-224"
        if model_id not in _loaded_models:
            print("Loading model and processor...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.float16).to(device)
            _loaded_models[model_id] = (processor, model)
        else:
            print("Reusing cached model and processor.")
            processor, model = _loaded_models[model_id]


    return processor, model

def four_position_pos(model_name, pos, ylim, colour=None):
   
    processor, model = load_model(model_name)

    count = 0
    item1_attn = []
    item2_attn = []
    item3_attn = []
    item4_attn = []
    baseline_attentions = []
    for i in range(100):
        image_path = f"4_shapes_same_dataset_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
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
       

        if target_colour == colour or colour is None:

            if pos == 'top_left':
                plotter.get_outputs(f"The figure is {top_left_colour}")
            elif pos == 'top_right':
                plotter.get_outputs(f"The figure is {top_right_colour}")
            elif pos == 'bottom_left':
                plotter.get_outputs(f"The figure is {bottom_left_colour}")
            elif pos == 'bottom_right':
                plotter.get_outputs(f"The figure is {bottom_right_colour}")
            elif pos == 'neg_top_left':
                plotter.get_outputs(f"The figure is not {top_left_colour}")
            elif pos == 'neg_top_right':
                plotter.get_outputs(f"The figure is not {top_right_colour}")
            elif pos == 'neg_bottom_left':
                plotter.get_outputs(f"The figure is not {bottom_left_colour}")
            elif pos == 'neg_bottom_right':
                plotter.get_outputs(f"The figure is not {bottom_right_colour}")

            print("outputs:", plotter.print_output())
            bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
            item1_attn.append(bbox_attentions[top_left_colour])
            item2_attn.append(bbox_attentions[top_right_colour])
            item3_attn.append(bbox_attentions[bottom_left_colour])
            item4_attn.append(bbox_attentions[bottom_right_colour])
            baseline_attentions.append(baseline_attn)
            count += 1

    print("finished processing images, now plotting averages...")

    # plot average attention across items
    avg_item1_attn = np.mean(np.array(item1_attn), axis=0)
    avg_item2_attn = np.mean(np.array(item2_attn), axis=0)
    avg_item3_attn = np.mean(np.array(item3_attn), axis=0)
    avg_item4_attn = np.mean(np.array(item4_attn), axis=0)
    avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)
    std_item1_attn = np.std(np.array(item1_attn), axis=0)
    std_item2_attn = np.std(np.array(item2_attn), axis=0)
    std_item3_attn = np.std(np.array(item3_attn), axis=0)
    std_item4_attn = np.std(np.array(item4_attn), axis=0)
    if model_name == 'llava':
        layers = list(range(32))
    elif model_name == 'internvl':
        layers = list(range(36))
    elif model_name == 'paligemma':
        layers = list(range(26))
    plt.figure(figsize=(10, 6))
    # plt.plot(layers, avg_item1_attn, label='Top Left Shape Attention')
    # plt.plot(layers, avg_item2_attn, label='Top Right Shape Attention')
    # plt.plot(layers, avg_item3_attn, label='Bottom Left Shape Attention')
    # plt.plot(layers, avg_item4_attn, label='Bottom Right Shape Attention')

    #plt.plot(layers, avg_baseline_attn, label='Baseline Attention', color='gray', linestyle='--')
    # plot mentioned shape and non-mentinoed shapes (averaged together)
    if pos in ['top_left', 'neg_top_left']:
        plt.plot(layers, avg_item1_attn, label='Top Left Shape Attention')
        avg_other_attn = np.mean(np.array([avg_item2_attn, avg_item3_attn, avg_item4_attn]), axis=0)
        plt.plot(layers, avg_other_attn, label='Other Shapes Attention', color='orange')
    elif pos in ['top_right', 'neg_top_right']:
        plt.plot(layers, avg_item2_attn, label='Top Right Shape Attention')
        avg_other_attn = np.mean(np.array([avg_item1_attn, avg_item3_attn, avg_item4_attn]), axis=0)
        plt.plot(layers, avg_other_attn, label='Other Shapes Attention', color='orange')
    elif pos in ['bottom_left', 'neg_bottom_left']:
        plt.plot(layers, avg_item3_attn, label='Bottom Left Shape Attention')
        avg_other_attn = np.mean(np.array([avg_item1_attn, avg_item2_attn, avg_item4_attn]), axis=0)
        plt.plot(layers, avg_other_attn, label='Other Shapes Attention', color='orange')
    elif pos in ['bottom_right', 'neg_bottom_right']:
        plt.plot(layers, avg_item4_attn, label='Bottom Right Shape Attention')
        avg_other_attn = np.mean(np.array([avg_item1_attn, avg_item2_attn, avg_item3_attn]), axis=0)
        plt.plot(layers, avg_other_attn, label='Other Shapes Attention', color='orange')
    




    # plot confidence intervals
    # plt.fill_between(layers, avg_item1_attn - std_item1_attn, avg_item1_attn + std_item1_attn, color='blue', alpha=0.2)
    # plt.fill_between(layers, avg_item2_attn - std_item2_attn, avg_item2_attn + std_item2_attn, color='red', alpha=0.2)    
    # plt.fill_between(layers, avg_item3_attn - std_item3_attn, avg_item3_attn + std_item3_attn, color='green', alpha=0.2)
    # plt.fill_between(layers, avg_item4_attn - std_item4_attn, avg_item4_attn + std_item4_attn, color='orange', alpha=0.2)
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title('Average Attention Scores Through Layers')
    plt.legend()
    # fix y axis limit
    plt.ylim(0, ylim)
    #plt.ylim(0, 0.01)
    #plt.ylim(0, 0.02)
    plt.savefig(f'plots/4_obj_plots_grouped/{model_name}_{pos}_{colour+"_" if colour else ""}all.png')
    print(f"Processed {count} images matching criteria.")

def get_pos_results(model_name, ylim, colour=None):
   
    processor, model = load_model(model_name)

    count = 0
    mentioned_attn = []
    not_mentioned_attn = []
    baseline_attentions = []
    for i in range(100):
        image_path = f"4_shapes_same_dataset_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
        top_left_colour = plotter.get_shape_by_position('top_left').colour
        top_right_colour = plotter.get_shape_by_position('top_right').colour
        bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
        bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour
        plotter.set_model(model, processor)

        for pos in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:

            if pos == 'top_left' or pos == 'neg_top_left':
                target_colour = top_left_colour
            elif pos == 'top_right' or pos == 'neg_top_right':
                target_colour = top_right_colour
            elif pos == 'bottom_left' or pos == 'neg_bottom_left':
                target_colour = bottom_left_colour
            elif pos == 'bottom_right' or pos == 'neg_bottom_right':
                target_colour = bottom_right_colour
    

            if pos == 'top_left':
                plotter.get_outputs(f"The figure is {top_left_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[top_left_colour])
                not_mentioned_attn.append(np.mean([bbox_attentions[colour] for colour in [top_right_colour, bottom_left_colour, bottom_right_colour]], axis=0))
            elif pos == 'top_right':
                plotter.get_outputs(f"The figure is {top_right_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[top_right_colour])
                not_mentioned_attn.append(np.mean([bbox_attentions[colour] for colour in [top_left_colour, bottom_left_colour, bottom_right_colour]], axis=0))
            elif pos == 'bottom_left':
                plotter.get_outputs(f"The figure is {bottom_left_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[bottom_left_colour])
                not_mentioned_attn.append(np.mean([bbox_attentions[colour] for colour in [top_left_colour, top_right_colour, bottom_right_colour]], axis=0))
            elif pos == 'bottom_right':
                plotter.get_outputs(f"The figure is {bottom_right_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[bottom_right_colour])
                not_mentioned_attn.append(np.mean([bbox_attentions[colour] for colour in [top_left_colour, top_right_colour, bottom_left_colour]], axis=0))
            elif pos == 'neg_top_left':
                plotter.get_outputs(f"The figure is not {top_left_colour}")
            elif pos == 'neg_top_right':
                plotter.get_outputs(f"The figure is not {top_right_colour}")
            elif pos == 'neg_bottom_left':
                plotter.get_outputs(f"The figure is not {bottom_left_colour}")
            elif pos == 'neg_bottom_right':
                plotter.get_outputs(f"The figure is not {bottom_right_colour}")

            print("outputs:", plotter.print_output())
            baseline_attentions.append(baseline_attn)
            count += 1

    print("finished processing images, now plotting averages...")

    # plot average attention across items
    aveg_mentioned_attn = np.mean(np.array(mentioned_attn), axis=0)
    avg_not_mentioned_attn = np.mean(np.array(not_mentioned_attn), axis=0)

    return aveg_mentioned_attn, avg_not_mentioned_attn

def get_neg_results(model_name, ylim, colour=None):
   
    processor, model = load_model(model_name)

    count = 0
    mentioned_attn = []
    not_mentioned_attn = []
    baseline_attentions = []
    for i in range(100):
        image_path = f"4_shapes_same_dataset_1000/images/image_{i:04d}.png"
        plotter = Plotter(image_path)
        top_left_colour = plotter.get_shape_by_position('top_left').colour
        top_right_colour = plotter.get_shape_by_position('top_right').colour
        bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
        bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour
        plotter.set_model(model, processor)

        for pos in ['neg_top_left', 'neg_top_right', 'neg_bottom_left', 'neg_bottom_right']:

            if pos == 'top_left' or pos == 'neg_top_left':
                target_colour = top_left_colour
            elif pos == 'top_right' or pos == 'neg_top_right':
                target_colour = top_right_colour
            elif pos == 'bottom_left' or pos == 'neg_bottom_left':
                target_colour = bottom_left_colour
            elif pos == 'bottom_right' or pos == 'neg_bottom_right':
                target_colour = bottom_right_colour
        


            if pos == 'neg_top_left':
                plotter.get_outputs(f"The figure is not {top_left_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[top_left_colour])
                not_mentioned_attn.append(np.mean([bbox_attentions[colour] for colour in [top_right_colour, bottom_left_colour, bottom_right_colour]], axis=0))
            elif pos == 'neg_top_right':
                plotter.get_outputs(f"The figure is not {top_right_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[top_right_colour])
                not_mentioned_attn.append(np.mean([bbox_attentions[colour] for colour in [top_left_colour, bottom_left_colour, bottom_right_colour]], axis=0))
            elif pos == 'neg_bottom_left':
                plotter.get_outputs(f"The figure is not {bottom_left_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[bottom_left_colour])
                not_mentioned_attn.append(np.mean([bbox_attentions[colour] for colour in [top_left_colour, top_right_colour, bottom_right_colour]], axis=0))
            elif pos == 'neg_bottom_right':
                plotter.get_outputs(f"The figure is not {bottom_right_colour}")
                bbox_attentions, baseline_attn = plotter.plot_attention_through_layers()
                mentioned_attn.append(bbox_attentions[bottom_right_colour])
                not_mentioned_attn.append(np.mean([bbox_attentions[colour] for colour in [top_left_colour, top_right_colour, bottom_left_colour]], axis=0))


            print("outputs:", plotter.print_output())
            baseline_attentions.append(baseline_attn)
            count += 1

    print("finished processing images, now plotting averages...")

    # plot average attention across items
    aveg_mentioned_attn = np.mean(np.array(mentioned_attn), axis=0)
    avg_not_mentioned_attn = np.mean(np.array(not_mentioned_attn), axis=0)
    avg_baseline_attn = np.mean(np.array(baseline_attentions), axis=0)

    return aveg_mentioned_attn, avg_not_mentioned_attn


def model_worker(model_name,  ylim, results_queue):
    """
    Handles all GPU logic for a single model and returns plot-ready data.
    """
    # 1. Run your existing extraction logic
    # Make sure these functions return numpy arrays/lists, not Torch tensors!
    pos_m, pos_nm = get_pos_results(model_name, ylim)
    neg_m, neg_nm = get_neg_results(model_name, ylim)
    
    def get_ci(data):
        """Calculate confidence interval for a list of values."""
        if len(data) == 0:
            return np.zeros_like(data)  # Return zeros if no data
        mean = np.mean(data, axis=0)
        sem = np.std(data, axis=0) / np.sqrt(len(data))
        z_score = 1.96  # For 95% confidence
        ci = z_score * sem
        return ci
    # 2. Package data to send back
    # We include everything needed for the plot (means and CIs)
    results = {
        "model_name": model_name,
        "pos": (pos_m, get_ci(pos_m), pos_nm, get_ci(pos_nm)), 
        "neg": (neg_m, get_ci(neg_m), neg_nm, get_ci(neg_nm)),

    }
    
    results_queue.put(results)

if __name__ == "__main__":


    models = ['llava', 'internvl', 'paligemma']
    ylims = [0.008, 0.005, 0.02]
    layers = [32, 36, 26]
    data_dict = {}

    # Sequential Multiprocessing: Process one model at a time to save VRAM
    for model_name, ylim in zip(models, ylims):
        print(f"Starting Process for: {model_name}")
        
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=model_worker, args=(model_name, ylim, q))
        
        p.start()
        data_dict[model_name] = q.get() # Grab the data
        p.join() # Process ends, GPU memory is fully cleared
        
        print(f"Memory cleared for {model_name}")

    display_names = {
        "llava": "LLaVA-1.5",
        "internvl": "InternVL3.5",
        "paligemma": "Paligemma2"
    }
    # --- Plotting Section (Runs in Main Process) ---
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    
    for i, model_name in enumerate(models):
        res = data_dict[model_name]
        num_layers = layers[i]
        x = list(range(num_layers))
        ylim = ylims[i]
        
        # Get the pretty name for the title
        pretty_name = display_names.get(model_name, model_name)

        # Helper to plot both rows efficiently
        for row, condition in enumerate(["pos", "neg"]):
            # Expecting your worker to return: (avg_m, ci_m, avg_nm, ci_nm)
            avg_m, ci_m, avg_nm, ci_nm = res[condition]
            
            ax = axs[row, i]
            
            # Plot Mentioned (Blue)
            ax.plot(x, avg_m, label='Mentioned Colour', color='blue', linewidth=2)
            ax.fill_between(x, avg_m - ci_m, avg_m + ci_m, color='blue', alpha=0.2)
            
            # Plot Not Mentioned (Orange)
            ax.plot(x, avg_nm, label='Not Mentioned Colour', color='red', linewidth=2)
            ax.fill_between(x, avg_nm - ci_nm, avg_nm + ci_nm, color='red', alpha=0.2)
            
            # Formatting
            cond_title = "Positive" if condition == "pos" else "Negative"
            ax.set_title(f"{pretty_name} - {cond_title}", fontsize=16, fontweight='bold')
            ax.set_xlabel("Layer", fontsize=16)
            ax.set_ylabel("Attention", fontsize=16)
            ax.set_ylim(0, ylim)
            ax.legend(loc='upper left', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig("plots/2d_datasets_2/grouped_4_shape_attention_new.png")
    plt.show()
