from plotter import Plotter
from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, InternVLForConditionalGeneration
from transformers import AutoModelForImageTextToText
import os 
import argparse
import multiprocessing as mp
import random
import pandas as pd

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


# write a function which tests how well a model understands negation
# we have images of two objects, firsly ask a positive question `Is the object on the left red?` and then a negative question `Is the object on the left not red?`
# I want to print out the accuracy for each of the three models on the positive and negative questions, and also print out some examples of where the model got it wrong for each question type.
def get_2_object_accuracy(model_name, data_dir, seed, prompt_num=0):
    print("in get 2 object accuracy")
    # set seed
    random.seed(seed)
    processor, model = load_model(model_name)

    # create dictionary that saves results for each mode for positive, negative, positive false and negative false questions
    results = {
        "positive": {"correct": 0, "total": 0, "examples": []},
        "negative": {"correct": 0, "total": 0, "examples": []},
        "positive_false": {"correct": 0, "total": 0, "examples": []},
        "negative_false": {"correct": 0, "total": 0, "examples": []},
    }

    question_types = ["positive", "negative", "positive_false", "negative_false"]
    answer_mapping = {
        "positive": True,
        "negative": True,
        "positive_false": False,
        "negative_false": False,
    }
    article = "a " if data_dir == "whatsup" else ""
    object_word = "" if data_dir == "whatsup" else "object"


    
    # load the data
    for image_file in os.listdir(os.path.join(data_dir, "images"))[0:100]:
        print("running for image ", image_file)
        image_path = os.path.join(data_dir, "images", image_file)
        print(f"Processing image: {image_path}")
        plotter = Plotter(image_path)
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour

        # randomly decide whether to ask about the left or right object first to avoid biasing the model
        pos1 = "left" if random.random() < 0.5 else "right"
        if pos1 == "left":
            col1, col2 = left_colour, right_colour
        else:
            col1, col2 = right_colour, left_colour

        templates = {
            "prompt_type_0": {
                "positive": f"Is this statement correct? The object on the {pos1} is {article}{col1}. Answer yes or no.",           # True: "Top left is Blue"
                "negative": f"Is this statement correct? The object on the {pos1} is not {article}{col2}. Answer yes or no.",       # True: "Top left is not Red"
                "positive_false": f"Is this statement correct? The object on the {pos1} is {article}{col2}. Answer yes or no.",     # False: "Top left is Red"
                "negative_false": f"Is this statement correct? The object on the {pos1} is not {article}{col1}. Answer yes or no."  # False: "Top left is not Blue"
            },
            "prompt_type_1": {
                "positive": f"Is the object on the {pos1} {article}{col1}? Answer yes or no.",           # True: "Top left is Blue"
                "negative": f"Is the object on the {pos1} not {article}{col2}? Answer yes or no.",       # True: "Top left is not Red"
                "positive_false": f"Is the object on the {pos1} {article}{col2}? Answer yes or no.",     # False: "Top left is Red"
                "negative_false": f"Is the object on the {pos1} not {article}{col1}? Answer yes or no."  # False: "Top left is not Blue"
            },
            "prompt_type_2": {
                "positive": f"Does this statement match the image? The object on the {pos1} is {article}{col1}. Answer yes or no.",           # True: "Top left is Blue"
                "negative": f"Does this statement match the image? The object on the {pos1} is not {article}{col2}. Answer yes or no.",       # True: "Top left is not Red"
                "positive_false": f"Does this statement match the image? The object on the {pos1} is {article}{col2}. Answer yes or no.",     # False: "Top left is Red"
                "negative_false": f"Does this statement match the image? The object on the {pos1} is not {article}{col1}. Answer yes or no."  # False: "Top left is not Blue"
            },
        }
        

        # get the ground truth for the image

        # ask the model the positive question
        plotter.set_model(model, processor)
        print(f"Seed: {seed} random: {random.random()}")
        for question_type in question_types:
            question = templates[f"prompt_type_{prompt_num}"][question_type]
            plotter.get_outputs(question)
            model_answer = plotter.print_output()
            model_answer_bool = model_answer.lower().strip() in ["yes", "true", "correct"]
            is_correct = model_answer_bool == answer_mapping[question_type]
            results[question_type]["total"] += 1
            if is_correct:
                results[question_type]["correct"] += 1
            else:
                results[question_type]["examples"].append((image_file, question, model_answer))

            print(f"Question: {question}")
            print(f"Model Answer: {model_answer}")
            print(f"Is Correct: {is_correct}")
    return results

def run_2_object_results():
    import json
    question_types = ["positive", "negative", "positive_false", "negative_false"]
    # creaet dictionary for each seeded set of results
    all_results = {}
    for i in range(3): # run with the three different prompt types
        all_results[i] = {}
        for seed in [0, 42, 100]:  # Run with different seeds for robustness
            all_results[i][seed] = {} 
            print(f"\nRunning with seed: {seed}")
            for model_name in ["llava", "internvl", "paligemma"]:
                print(f"Evaluating model: {model_name}")
                # use multiprocessing to avoid cuda crash and collect all results
                with mp.Pool(processes=1) as pool:
                    results = pool.apply(get_2_object_accuracy, args=(model_name, "2d_dataset_fixed_positions_1000", seed, i))
                print(f"Results for {model_name}:")
                for question_type, result in results.items():
                    accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                    print(f"{question_type.capitalize()} Accuracy: {accuracy:.2f}%")
                    print(f"Examples where the model got it wrong for {question_type}:")
                    for example in result["examples"][:5]:  # print first 5 examples
                        print(f"Model: {model_name}, Image: {example[0]}, Question: {example[1]}, Model Answer: {example[2]}")
                all_results[i][seed][model_name] = results

    # save results to a json file
    with open("2_object_results.json", "w") as f:
        json.dump(all_results, f, indent=4)


    print("Saved results to 2_object_results.json")
    print_2_obj_results()

def run_whatsup_results():
    import json
    question_types = ["positive", "negative", "positive_false", "negative_false"]
    # creaet dictionary for each seeded set of results
    all_results = {}
    for i in range(3): # run with the three different prompt types
        all_results[i] = {}
        for seed in [0, 42, 100]:  # Run with different seeds for robustness
            all_results[i][seed] = {} 
            print(f"\nRunning with seed: {seed}")
            for model_name in ["llava", "internvl", "paligemma"]:
                print(f"Evaluating model: {model_name}")
                # use multiprocessing to avoid cuda crash and collect all results
                with mp.Pool(processes=1) as pool:
                    results = pool.apply(get_2_object_accuracy, args=(model_name, "whatsup", seed, i))
                print(f"Results for {model_name}:")
                for question_type, result in results.items():
                    accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                    print(f"{question_type.capitalize()} Accuracy: {accuracy:.2f}%")
                    print(f"Examples where the model got it wrong for {question_type}:")
                    for example in result["examples"][:5]:  # print first 5 examples
                        print(f"Model: {model_name}, Image: {example[0]}, Question: {example[1]}, Model Answer: {example[2]}")
                all_results[i][seed][model_name] = results

    # save results to a json file
    with open("whatsup_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("Saved results to whatsup_results.json")
    print_whatsup_results()

def get_4_object_accuracy(model_name, data_dir, seed, prompt_num=0):
    # set seed
    random.seed(seed)
    processor, model = load_model(model_name)

    # create dictionary that saves results for each mode for positive, negative, positive false and negative false questions
    results = {
        "positive": {"correct": 0, "total": 0, "examples": []},
        "negative": {"correct": 0, "total": 0, "examples": []},
        "positive_false": {"correct": 0, "total": 0, "examples": []},
        "negative_false": {"correct": 0, "total": 0, "examples": []},
    }

    question_types = ["positive", "negative", "positive_false", "negative_false"]
    answer_mapping = {
        "positive": True,
        "negative": True,
        "positive_false": False,
        "negative_false": False,
    }
    # load the data
    for image_file in os.listdir(os.path.join(data_dir, "images"))[0:100]:
        if image_file.endswith(".png"):
            image_path = os.path.join(data_dir, "images", image_file)
            print(f"Processing image: {image_path}")
            plotter = Plotter(image_path)
            # get the ground truth for the image
            top_left_colour = plotter.get_shape_by_position('top_left').colour
            top_right_colour = plotter.get_shape_by_position('top_right').colour
            bottom_left_colour = plotter.get_shape_by_position('bottom_left').colour
            bottom_right_colour = plotter.get_shape_by_position('bottom_right').colour

            shapes = {
                "top left": top_left_colour,
                "top right": top_right_colour,
                "bottom left": bottom_left_colour,
                "bottom right": bottom_right_colour
            }

            # 2. Pick a primary subject and a secondary (different) subject for comparisons
            pos1, pos2 = random.sample(list(shapes.keys()), 2)
            col1, col2 = shapes[pos1], shapes[pos2]

            # 3. Use a dictionary of templates to keep things organized
            templates = {
                "prompt_type_0": {
                    "positive": f"The object at the {pos1} is {col1}.",           # True: "Top left is Blue"
                    "negative": f"The object at the {pos1} is not {col2}.",       # True: "Top left is not Red"
                    "positive_false": f"The object at the {pos1} is {col2}.",     # False: "Top left is Red"
                    "negative_false": f"The object at the {pos1} is not {col1}."  # False: "Top left is not Blue"
                },
                "prompt_type_1": {
                    "positive": f"Is the object on the {pos1} {col1}? Answer yes or no.",           # True: "Top left is Blue"
                    "negative": f"Is the object on the {pos1} not {col2}? Answer yes or no.",       # True: "Top left is not Red"
                    "positive_false": f"Is the object on the {pos1} {col2}? Answer yes or no.",     # False: "Top left is Red"
                    "negative_false": f"Is the object on the {pos1} not {col1}? Answer yes or no."  # False: "Top left is not Blue"
                },
                "prompt_type_2": {
                    "positive": f"Is there a {col1} object on the {pos1}? Answer yes or no.",           # True: "Top left is Blue"
                    "negative": f"Is there a not {col2} object on the {pos1}? Answer yes or no.",       # True: "Top left is not Red"
                    "positive_false": f"Is there a {col2} object on the {pos1}? Answer yes or no.",     # False: "Top left is Red"
                    "negative_false": f"Is there a not {col1} object on the {pos1}? Answer yes or no."  # False: "Top left is not Blue"
                }
            }
            # ask the model the positive question
            plotter.set_model(model, processor)
            for question_type in question_types:
                question = templates[f"prompt_type_{prompt_num}"][question_type]
                print(question)
                plotter.get_outputs(question)
                model_answer = plotter.print_output()
                print(f"Answer: {model_answer}")
                model_answer_bool = model_answer.lower().strip() in ["yes", "true", "correct"]
                is_correct = model_answer_bool == answer_mapping[question_type]
                print(f"Is the model correct? {is_correct}")
                results[question_type]["total"] += 1
                if is_correct:
                    results[question_type]["correct"] += 1
                else:
                    results[question_type]["examples"].append((image_file, question, model_answer))


    return results

def run_4_object_results():
    import json
    question_types = ["positive", "negative", "positive_false", "negative_false"]
    # creaet dictionary for each seeded set of results
    all_results = {}
    for i in range(3): # run with the three different prompt types
        all_results[i] = {}
        for seed in [0, 42, 100]:  # Run with different seeds for robustness
            all_results[i][seed] = {} 
            print(f"\nRunning with seed: {seed}")
            for model_name in ["llava", "internvl", "paligemma"]:
                print(f"Evaluating model: {model_name}")
                # use multiprocessing to avoid cuda crash and collect all results
                with mp.Pool(processes=1) as pool:
                    results = pool.apply(get_4_object_accuracy, args=(model_name, "4_shapes_same_dataset_1000", seed, i))
                print(f"Results for {model_name}:")
                for question_type, result in results.items():
                    accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                    print(f"{question_type.capitalize()} Accuracy: {accuracy:.2f}%")
                    print(f"Examples where the model got it wrong for {question_type}:")
                    for example in result["examples"][:5]:  # print first 5 examples
                        print(f"Model: {model_name}, Image: {example[0]}, Question: {example[1]}, Model Answer: {example[2]}")
                all_results[i][seed][model_name] = results

    # save results to a json file
    with open("4_object_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("Saved results to 4_object_results.json")
    print_results("4_object")

def print_2_obj_results():
    import json
    with open("2_object_results.json", "r") as f:
        all_results = json.load(f)

    # prompt_type, seed, model, question_type, correct

    for ptype in sorted(all_results.keys()):
        print(f"\nPrompt Type: {ptype}")
        for model in ["llava", "internvl", "paligemma"]:
            print(f"\nModel: {model}")
            for question_type in ["positive", "negative", "positive_false", "negative_false"]:
                accuracies = []
                pos_accuracies = []
                neg_accuracies = []
                for seed in [0, 42, 100]:
                    result = all_results[ptype][str(seed)][model][question_type]
                    accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                    print(f"  Prompt type {ptype} , model {model}, seed {seed}: {accuracy:.2f}%")
                    accuracies.append(accuracy)
                avg_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                if question_type in ["positive", "positive_false"]:
                    pos_accuracies.append(avg_accuracy)
                else:
                    neg_accuracies.append(avg_accuracy)
                print(f"{question_type.capitalize()} Accuracy: {avg_accuracy:.2f}% (±{std_accuracy:.2f})")


def print_whatsup_results():
    import json
    with open("whatsup_results.json", "r") as f:
        all_results = json.load(f)

    # prompt_type, seed, model, question_type, correct

    for ptype in sorted(all_results.keys()):
        print(f"\nPrompt Type: {ptype}")
        for model in ["llava", "internvl", "paligemma"]:
            print(f"\nModel: {model}")
            for question_type in ["positive", "negative", "positive_false", "negative_false"]:
                accuracies = []
                for seed in [0, 42, 100]:
                    result = all_results[ptype][str(seed)][model][question_type]
                    accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                    print(f"  Prompt type {ptype} , model {model}, seed {seed}: {accuracy:.2f}%")
                    accuracies.append(accuracy)
                avg_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                print(f"{question_type.capitalize()} Accuracy: {avg_accuracy:.2f}% (±{std_accuracy:.2f})")

def print_results(results_type):
    import json

    if results_type == "2_object":
        with open("2_object_results.json", "r") as f:
            all_results = json.load(f)
    elif results_type == "whatsup":
        with open("whatsup_results.json", "r") as f:
            all_results = json.load(f)
    elif results_type == "4_object":
        with open("4_object_results.json", "r") as f:
            all_results = json.load(f)

    # prompt_type, seed, model, question_type, correct

    for ptype in sorted(all_results.keys()):
        print(f"\nPrompt Type: {ptype}")
        for model in ["llava", "internvl", "paligemma"]:
            print(f"\nModel: {model}")
            pos_accuracies = []
            neg_accuracies = []
            for question_type in ["positive", "negative", "positive_false", "negative_false"]:
                accuracies = []
                for seed in [0, 42, 100]:
                    result = all_results[ptype][str(seed)][model][question_type]
                    accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                    print(f"  Prompt type {ptype} , model {model}, seed {seed}: {accuracy:.2f}%")
                    accuracies.append(accuracy)
                    if question_type in ["positive", "positive_false"]:
                        pos_accuracies.append(accuracy)
                    else:
                        neg_accuracies.append(accuracy)
                avg_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)

                print(f"{question_type.capitalize()} Accuracy: {avg_accuracy:.2f}% (±{std_accuracy:.2f})")
                # print positive and negative accuracy 
            print(f" Model: {model}, prompt type {ptype} Positive Accuracy: {np.mean(pos_accuracies):.2f}% (±{np.std(pos_accuracies):.2f}%)")
            print(f" Model: {model}, prompt type {ptype} Negative Accuracy: {np.mean(neg_accuracies):.2f}% (±{np.std(neg_accuracies):.2f}%)")

def print_overall_pos_v_neg():
    import json


    for file in ["2_object_results.json", "whatsup_results.json", "4_object_results.json"]:
        print(f"\nResults for {file}:")
        with open(file, "r") as f:
            all_results = json.load(f)
        results = {}
        for ptype in sorted(all_results.keys()):
            for model in ["llava", "internvl", "paligemma"]:
                pos_accuracies = []
                neg_accuracies = []
                for question_type in ["positive", "negative", "positive_false", "negative_false"]:
                    for seed in [0, 42, 100]:
                        result = all_results[ptype][str(seed)][model][question_type]
                        accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                        if question_type in ["positive", "positive_false"]:
                            pos_accuracies.append(accuracy)
                        else:
                            neg_accuracies.append(accuracy)
                avg_pos_accuracy = np.mean(pos_accuracies)
                avg_neg_accuracy = np.mean(neg_accuracies)

                if model not in results:
                    results[model] = {"positive": [], "negative": []}
                results[model]["positive"].append(avg_pos_accuracy)
                results[model]["negative"].append(avg_neg_accuracy)

        for model, acc in results.items():
            print(f"Model: {model}")
            print(f"  Average Positive Accuracy across datasets: {np.mean(acc['positive']):.2f}% (±{np.std(acc['positive']):.2f}%)")
            print(f"  Average Negative Accuracy across datasets: {np.mean(acc['negative']):.2f}% (±{np.std(acc['negative']):.2f}%)")

if __name__ == "__main__":
    #run_whatsup_results()
    #run_2_object_results()
    #run_4_object_results()
    #get_2_object_accuracy("llava", "whatsup", 42, 0)
    print_overall_pos_v_neg()