from plotter import Plotter
from functions import load_model
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
    for image_file in os.listdir(os.path.join(data_dir, "images"))[0:10]:
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
    mp.set_start_method('spawn', force=True)
    for i in range(3): # run with the three different prompt types
        all_results[i] = {}
        for seed in [0, 42, 100]:  # Run with different seeds for robustness
            all_results[i][seed] = {} 
            print(f"\nRunning with seed: {seed}")
            for model_name in ["llava", "internvl", "paligemma"]:
                print(f"Evaluating model: {model_name}")
                # use multiprocessing to avoid cuda crash and collect all results
                with mp.Pool(processes=1) as pool:
                    results = pool.apply(get_2_object_accuracy, args=(model_name, "binary", seed, i))
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
    mp.set_start_method('spawn', force=True)
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
    for image_file in os.listdir(os.path.join(data_dir, "images"))[0:10]:
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
    mp.set_start_method('spawn', force=True)
    for i in range(3): # run with the three different prompt types
        all_results[i] = {}
        for seed in [0, 42, 100]:  # Run with different seeds for robustness
            all_results[i][seed] = {} 
            print(f"\nRunning with seed: {seed}")
            for model_name in ["llava", "internvl", "paligemma"]:
                print(f"Evaluating model: {model_name}")
                # use multiprocessing to avoid cuda crash and collect all results
                with mp.Pool(processes=1) as pool:
                    results = pool.apply(get_4_object_accuracy, args=(model_name, "multary", seed, i))
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

    overall_restuls = {
        "llava": {"positive": [], "negative": []},
        "internvl": {"positive": [], "negative": []},
        "paligemma": {"positive": [], "negative": []},
    }

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
                            overall_restuls[model]["positive"].append(accuracy)
                        else:
                            neg_accuracies.append(accuracy)
                            overall_restuls[model]["negative"].append(accuracy)
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


    print("\nOverall Results Across All Datasets:")
    for model, acc in overall_restuls.items():
        print(f"Model: {model}")
        print(f"  Overall Positive Accuracy: {np.mean(acc['positive']):.2f}% (±{np.std(acc['positive']):.2f}%)")
        print(f"  Overall Negative Accuracy: {np.mean(acc['negative']):.2f}% (±{np.std(acc['negative']):.2f}%)")


def print_breakdown():
    print("In print breakdown")
    import json
    with open("2_object_results.json", "r") as f:
        all_results_2_obj = json.load(f)
    with open("whatsup_results.json", "r") as f:
        all_results_whatsup = json.load(f)
    with open("4_object_results.json", "r") as f:
        all_results_4_obj = json.load(f)


    overall_results = {
        "llava": {"positive": [], "negative": []},
        "internvl": {"positive": [], "negative": []},
        "paligemma": {"positive": [], "negative": []},
    }

    model_total_pos_accuracies = []
    model_total_neg_accuracies = []
    model_total_pos_false_accuracies = []
    model_total_neg_false_accuracies = []
    llava_results_pos = []
    llava_results_neg = []
    llava_results_pos_false = []
    llava_results_neg_false = []
    internvl_results_pos = []
    internvl_results_neg = []
    internvl_results_pos_false = []
    internvl_results_neg_false = []
    paligemma_results_pos = []
    paligemma_results_neg = []
    paligemma_results_pos_false = []
    paligemma_results_neg_false = []
    for filename in ["2_object_results.json", "whatsup_results.json", "4_object_results.json"]:
        with open(filename, "r") as f:
            results_loaded = json.load(f)
        dataset_accuracy_pos = []
        dataset_accuracy_neg = []
        for model in ["llava", "internvl", "paligemma"]:
            print("Model: ", model)
            model_level_pos_accuracies = []
            model_level_neg_accuracies = []
            model_level_pos_false_accuracies = []
            model_level_neg_false_accuracies = []
            # find result for each prompt and seed, print out then print average for prompt then print total average
            for ptype in sorted(results_loaded.keys()):
                promt_type_results_pos = []
                prompt_type_results_neg = []
                prompt_type_results_pos_false = []
                prompt_type_results_neg_false = []
                for seed in [0, 42, 100]:
                    result = results_loaded[ptype][str(seed)][model]
                    seed_level_accuracy_pos = []
                    seed_level_accuracy_neg = []
                    seed_level_accuracy_pos_false = []
                    seed_level_accuracy_neg_false = []
                    for question_type in ["positive", "negative","positive_false", "negative_false"]:
                        accuracy = (result[question_type]["correct"] / result[question_type]["total"] * 100) if result[question_type]["total"] > 0 else 0
                        #print("                    Prompt type: ", ptype, "seed: ", seed, "question type: ", question_type, "accuracy: ", accuracy)
                        if question_type == "positive":
                            overall_results[model]["positive"].append(accuracy)
                            seed_level_accuracy_pos.append(accuracy)
                            promt_type_results_pos.append(accuracy)
                            model_level_pos_accuracies.append(accuracy)
                            if model == "llava":
                                llava_results_pos.append(accuracy)
                            elif model == "internvl":
                                internvl_results_pos.append(accuracy)
                            elif model == "paligemma":
                                paligemma_results_pos.append(accuracy)
                        elif question_type == "negative":
                            overall_results[model]["negative"].append(accuracy)
                            seed_level_accuracy_neg.append(accuracy)
                            prompt_type_results_neg.append(accuracy)
                            model_level_neg_accuracies.append(accuracy)
                            if model == "llava":
                                llava_results_neg.append(accuracy)
                            elif model == "internvl":
                                internvl_results_neg.append(accuracy)
                            elif model == "paligemma":
                                paligemma_results_neg.append(accuracy)
                        elif question_type == "positive_false":
                            model_level_pos_false_accuracies.append(accuracy)
                            seed_level_accuracy_pos_false.append(accuracy)
                            prompt_type_results_pos_false.append(accuracy)
                            if model == "llava":
                                llava_results_pos_false.append(accuracy)
                            elif model == "internvl":
                                internvl_results_pos_false.append(accuracy)
                            elif model == "paligemma":
                                paligemma_results_pos_false.append(accuracy)
                        elif question_type == "negative_false":
                            model_level_neg_false_accuracies.append(accuracy)
                            seed_level_accuracy_neg_false.append(accuracy)
                            prompt_type_results_neg_false.append(accuracy)
                            if model == "llava":
                                llava_results_neg_false.append(accuracy)
                            elif model == "internvl":
                                internvl_results_neg_false.append(accuracy)
                            elif model == "paligemma":
                                paligemma_results_neg_false.append(accuracy)

                    #print("Model: ", model, ", Prompt type: ", ptype, ", seed: ", seed, ", positive accuracy: ", np.mean(seed_level_accuracy_pos), "std: ", np.std(seed_level_accuracy_pos), "negative accuracy: ", np.mean(seed_level_accuracy_neg), "std: ", np.std   (seed_level_accuracy_neg))
                    #print(f"Model: {model}, Prompt type: {ptype}, seed: {seed}, positive false accuracy: {np.mean(seed_level_accuracy_pos_false):.2f}% (±{np.std(seed_level_accuracy_pos_false):.2f}%)")
                print("Prompt type: ", ptype, ", positive accuracy: ", np.mean(promt_type_results_pos), "std: ", np.std(promt_type_results_pos), "negative accuracy: ", np.mean(prompt_type_results_neg), "std: ", np.std(prompt_type_results_neg))
                print("Prompt type: ", ptype, ", positive false accuracy: ", np.mean(prompt_type_results_pos_false), "std: ", np.std(prompt_type_results_pos_false), "negative false accuracy: ", np.mean(prompt_type_results_neg_false), "std: ", np.std(prompt_type_results_neg_false))
            print(f"dataset{filename}, {model} positive accuracy: {np.mean(model_level_pos_accuracies):.2f}% (±{np.std(model_level_pos_accuracies):.2f}%)")
            print(f"dataset{filename} {model} negative accuracy: {np.mean(model_level_neg_accuracies):.2f}% (±{np.std(model_level_neg_accuracies):.2f}%)")  
            print(f"dataset{filename}, {model} positive false accuracy: {np.mean(model_level_pos_false_accuracies):.2f}% (±{np.std(model_level_pos_false_accuracies):.2f}%)")
            print(f"dataset{filename} {model} negative false accuracy: {np.mean(model_level_neg_false_accuracies):.2f}% (±{np.std(model_level_neg_false_accuracies):.2f}%)")    

    print("Llava overall positive accuracy: ", np.mean(llava_results_pos), "std: ", np.std(llava_results_pos))
    print("Llava overall negative accuracy: ", np.mean(llava_results_neg), "std: ", np.std(llava_results_neg))
    print("InternVL overall positive accuracy: ", np.mean(internvl_results_pos), "std: ", np.std(internvl_results_pos))
    print("InternVL overall negative accuracy: ", np.mean(internvl_results_neg), "std: ", np.std(internvl_results_neg))
    print("Paligemma overall positive accuracy: ", np.mean(paligemma_results_pos), "std: ", np.std(paligemma_results_pos))
    print("Paligemma overall negative accuracy: ", np.mean(paligemma_results_neg), "std: ", np.std(paligemma_results_neg))
    print("Llava overall positive false accuracy: ", np.mean(llava_results_pos_false), "std: ", np.std(llava_results_pos_false))
    print("Llava overall negative false accuracy: ", np.mean(llava_results_neg_false), "std: ", np.std(llava_results_neg_false))
    print("InternVL overall positive false accuracy: ", np.mean(internvl_results_pos_false), "std: ", np.std(internvl_results_pos_false))
    print("InternVL overall negative false accuracy: ", np.mean(internvl_results_neg_false), "std: ", np.std(internvl_results_neg_false))
    print("Paligemma overall positive false accuracy: ", np.mean(paligemma_results_pos_false), "std: ", np.std(paligemma_results_pos_false))
    print("Paligemma overall negative false accuracy: ", np.mean(paligemma_results_neg_false), "std: ", np.std(paligemma_results_neg_false))

    print("=======================")
    print("Overall positive accuracy llava:", (np.mean(llava_results_pos)+np.mean(llava_results_pos_false))/2)
    print("Overall negative accuracy llava:", (np.mean(llava_results_neg)+np.mean(llava_results_neg_false))/2)
    print("Overall positive accuracy internvl:", (np.mean(internvl_results_pos)+np.mean(internvl_results_pos_false))/2)
    print("Overall negative accuracy internvl:", (  np.mean(internvl_results_neg)+np.mean(internvl_results_neg_false))/2)
    print("Overall positive accuracy paligemma:", (np.mean(paligemma_results_pos)+np.mean(paligemma_results_pos_false))/2)
    print("Overall negative accuracy paligemma:", (np.mean(paligemma_results_neg)+np.mean(paligemma_results_neg_false))/2)


def print_dataset_results_breakdown():
    import json
    for filename in ["2_object_results.json", "whatsup_results.json", "4_object_results.json"]:
        with open(filename, "r") as f:
            results_loaded = json.load(f)
        dataset_accuracy_pos = []
        dataset_accuracy_neg = []
        for model in ["llava", "internvl", "paligemma"]:
            print("Model: ", model)
            model_level_pos_accuracies = []
            model_level_neg_accuracies = []
            # find result for each prompt and seed, print out then print average for prompt then print total average
            for ptype in sorted(results_loaded.keys()):
                promt_type_results_pos = []
                prompt_type_results_neg = []
                for seed in [0, 42, 100]:
                    result = results_loaded[ptype][str(seed)][model]
                    seed_level_accuracy_pos = []
                    seed_level_accuracy_neg = []
                    for question_type in ["positive", "negative","positive_false", "negative_false"]:
                        accuracy = (result[question_type]["correct"] / result[question_type]["total"] * 100) if result[question_type]["total"] > 0 else 0
                        print("                    Prompt type: ", ptype, "seed: ", seed, "question type: ", question_type, "accuracy: ", accuracy)
                        if question_type in ["positive", "positive_false"]:
                            model_level_pos_accuracies.append(accuracy)
                        else:
                            model_level_neg_accuracies.append(accuracy)

            print("dataset: ", filename, ", model: ", model, ", positive accuracy: ", np.mean(model_level_pos_accuracies), "std: ", np.std(model_level_pos_accuracies), "negative accuracy: ", np.mean(model_level_neg_accuracies), "std: ", np.std(model_level_neg_accuracies))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Get downstream results for models")
      
  parser.add_argument(
      "--function", 
      type=str, 
      default="print", 
      help="Where to generate (gen) or print (print) the results"
  )

  args = parser.parse_args()

  if args.function == "gen":
      run_whatsup_results()
      run_2_object_results()
      run_4_object_results()