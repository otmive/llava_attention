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
def get_2_object_accuracy(model_name, data_dir, seed):
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
    # load the data
    for image_file in os.listdir(os.path.join(data_dir, "images"))[0:100]:
        print("running for image ", image_file)

        image_path = os.path.join(data_dir, "images", image_file)
        print(f"Processing image: {image_path}")
        plotter = Plotter(image_path)
        # get the ground truth for the image
        left_colour = plotter.get_left_shapes()[0].colour
        right_colour = plotter.get_right_shapes()[0].colour
        # ask the model the positive question
        plotter.set_model(model, processor)
        print(f"Seed: {seed} random: {random.random()}")
        for question_type in question_types:
            if question_type == "positive":
                if random.random() < 0.5:
                    question = f"Is this sentence correct? The object on the left is {article}{left_colour}. Answer yes or no."
                else:
                    question = f"Is this sentence correct? The object on the right is {article}{right_colour}. Answer yes or no."
            elif question_type == "negative":
                if random.random() < 0.5:
                    question = f"Is this sentence correct? The object on the right is not {article}{left_colour}. Answer yes or no."
                else:
                    question = f"Is this sentence correct? The object on the left is not {article}{right_colour}. Answer yes or no."
            elif question_type == "positive_false":
                if random.random() < 0.5:
                    question = f"Is this sentence correct? The object on the right is {article}{left_colour}. Answer yes or no."
                else:
                    question = f"Is this sentence correct? The object on the left is {article}{right_colour}. Answer yes or no."
            elif question_type == "negative_false":
                if random.random() < 0.5:
                    question = f"Is this sentence correct? The object on the left is not {article}{left_colour}. Answer yes or no."
                else:
                    question = f"Is this sentence correct? The object on the right is not {article}{right_colour}. Answer yes or no."
            
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
    for seed in [42, 43, 44]:  # Run with different seeds for robustness
        all_results[seed] = {} 
        print(f"\nRunning with seed: {seed}")
        for model_name in ["llava", "internvl", "paligemma"]:
            print(f"Evaluating model: {model_name}")
            # use multiprocessing to avoid cuda crash and collect all results
            with mp.Pool(processes=1) as pool:
                results = pool.apply(get_2_object_accuracy, args=(model_name, "2d_dataset_fixed_positions_1000", 42))
            print(f"Results for {model_name}:")
            for question_type, result in results.items():
                accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                print(f"{question_type.capitalize()} Accuracy: {accuracy:.2f}%")
                print(f"Examples where the model got it wrong for {question_type}:")
                for example in result["examples"][:5]:  # print first 5 examples
                    print(f"Model: {model_name}, Image: {example[0]}, Question: {example[1]}, Model Answer: {example[2]}")
            all_results[seed][model_name] = results

    # save results to a json file
    with open("2_object_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # print out results in a nice format
    for seed, models in all_results.items():
        print(f"\nSeed: {seed}")
        for model_name, results in models.items():
            print(f"Model: {model_name}")
            for question_type, result in results.items():
                accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                print(f"{question_type.capitalize()} Accuracy: {accuracy:.2f}%")
                print(f"Examples where the model got it wrong for {question_type}:")
                for example in result["examples"][:5]:  # print first 5 examples
                    print(f"Model: {model_name}, Image: {example[0]}, Question: {example[1]}, Model Answer: {example[2]}")

    # show results in a table with pandas, report average and standard deviation across seeds for each model and question type
    # report just average across seeds for each model and question type
    summary_data = []
    for model_name in ["llava", "internvl", "paligemma"]:
        for question_type in question_types:
            accuracies = []
            for seed in [42, 43, 44]:
                result = all_results[seed][model_name][question_type]
                accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                accuracies.append(accuracy)
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            summary_data.append({
                "Model": model_name,
                "Question Type": question_type,
                "Average Accuracy": avg_accuracy,
                "Std Accuracy": std_accuracy
            })
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary of Results:")
    print(summary_df)
    summary_df.to_csv("2_object_results_summary.csv", index=False)

    # group by positive and positive false together and group negative and negative false together and report average and standard deviation for each model
    grouped_summary_data = []
    for model_name in ["llava", "internvl", "paligemma"]:
        for group in ["positive", "negative"]:
            accuracies = []
            for seed in [42, 43, 44]:
                if group == "positive":
                    result_pos = all_results[seed][model_name]["positive"]
                    result_pos_false = all_results[seed][model_name]["positive_false"]
                    accuracy_pos = (result_pos["correct"] / result_pos["total"] * 100) if result_pos["total"] > 0 else 0
                    accuracy_pos_false = (result_pos_false["correct"] / result_pos_false["total"] * 100) if result_pos_false["total"] > 0 else 0
                    accuracies.append((accuracy_pos + accuracy_pos_false) / 2)
                else:
                    result_neg = all_results[seed][model_name]["negative"]
                    result_neg_false = all_results[seed][model_name]["negative_false"]
                    accuracy_neg = (result_neg["correct"] / result_neg["total"] * 100) if result_neg["total"] > 0 else 0
                    accuracy_neg_false = (result_neg_false["correct"] / result_neg_false["total"] * 100) if result_neg_false["total"] > 0 else 0
                    accuracies.append((accuracy_neg + accuracy_neg_false) / 2)
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            grouped_summary_data.append({
                "Model": model_name,
                "Question Group": group,
                "Average Accuracy": avg_accuracy,
                "Std Accuracy": std_accuracy
            })
    grouped_summary_df = pd.DataFrame(grouped_summary_data)
    print("\nGrouped Summary of Results:")
    print(grouped_summary_df)
    grouped_summary_df.to_csv("2d_dataset_grouped_summary.csv")

def run_whatsup_results():
    import json
    question_types = ["positive", "negative", "positive_false", "negative_false"]
    # creaet dictionary for each seeded set of results
    all_results = {}
    for seed in [42, 43, 44]:  # Run with different seeds for robustness
        all_results[seed] = {} 
        print(f"\nRunning with seed: {seed}")
        for model_name in ["llava", "internvl", "paligemma"]:
            print(f"Evaluating model: {model_name}")
            # use multiprocessing to avoid cuda crash and collect all results
            with mp.Pool(processes=1) as pool:
                results = pool.apply(get_2_object_accuracy, args=(model_name, "whatsup", 42))
            print(f"Results for {model_name}:")
            for question_type, result in results.items():
                accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                print(f"{question_type.capitalize()} Accuracy: {accuracy:.2f}%")
                print(f"Examples where the model got it wrong for {question_type}:")
                for example in result["examples"][:5]:  # print first 5 examples
                    print(f"Model: {model_name}, Image: {example[0]}, Question: {example[1]}, Model Answer: {example[2]}")
            all_results[seed][model_name] = results

    # save results to a json file
    with open("whatsup_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # print out results in a nice format
    for seed, models in all_results.items():
        print(f"\nSeed: {seed}")
        for model_name, results in models.items():
            print(f"Model: {model_name}")
            for question_type, result in results.items():
                accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                print(f"{question_type.capitalize()} Accuracy: {accuracy:.2f}%")
                print(f"Examples where the model got it wrong for {question_type}:")
                for example in result["examples"][:5]:  # print first 5 examples
                    print(f"Model: {model_name}, Image: {example[0]}, Question: {example[1]}, Model Answer: {example[2]}")

    # show results in a table with pandas, report average and standard deviation across seeds for each model and question type
    # report just average across seeds for each model and question type
    summary_data = []
    for model_name in ["llava", "internvl", "paligemma"]:
        for question_type in question_types:
            accuracies = []
            for seed in [42, 43, 44]:
                result = all_results[seed][model_name][question_type]
                accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                accuracies.append(accuracy)
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            summary_data.append({
                "Model": model_name,
                "Question Type": question_type,
                "Average Accuracy": avg_accuracy,
                "Std Accuracy": std_accuracy
            })
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary of Results:")
    print(summary_df)
    summary_df.to_csv("whatsup_results_summary.csv", index=False)

    # group by positive and positive false together and group negative and negative false together and report average and standard deviation for each model
    grouped_summary_data = []
    for model_name in ["llava", "internvl", "paligemma"]:
        for group in ["positive", "negative"]:
            accuracies = []
            for seed in [42, 43, 44]:
                if group == "positive":
                    result_pos = all_results[seed][model_name]["positive"]
                    result_pos_false = all_results[seed][model_name]["positive_false"]
                    accuracy_pos = (result_pos["correct"] / result_pos["total"] * 100) if result_pos["total"] > 0 else 0
                    accuracy_pos_false = (result_pos_false["correct"] / result_pos_false["total"] * 100) if result_pos_false["total"] > 0 else 0
                    accuracies.append((accuracy_pos + accuracy_pos_false) / 2)
                else:
                    result_neg = all_results[seed][model_name]["negative"]
                    result_neg_false = all_results[seed][model_name]["negative_false"]
                    accuracy_neg = (result_neg["correct"] / result_neg["total"] * 100) if result_neg["total"] > 0 else 0
                    accuracy_neg_false = (result_neg_false["correct"] / result_neg_false["total"] * 100) if result_neg_false["total"] > 0 else 0
                    accuracies.append((accuracy_neg + accuracy_neg_false) / 2)
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            grouped_summary_data.append({
                "Model": model_name,
                "Question Group": group,
                "Average Accuracy": avg_accuracy,
                "Std Accuracy": std_accuracy
            })
    grouped_summary_df = pd.DataFrame(grouped_summary_data)
    print("\nGrouped Summary of Results:")
    print(grouped_summary_df)
    grouped_summary_df.to_csv("whatsup_grouped_summary.csv")

def get_4_object_accuracy(model_name, data_dir, seed):
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
                "positive": f"The object at the {pos1} is {col1}.",           # True: "Top left is Blue"
                "negative": f"The object at the {pos1} is not {col2}.",       # True: "Top left is not Red"
                "positive_false": f"The object at the {pos1} is {col2}.",     # False: "Top left is Red"
                "negative_false": f"The object at the {pos1} is not {col1}."  # False: "Top left is not Blue"
            }
            # ask the model the positive question
            plotter.set_model(model, processor)
            for question_type in question_types:
                statement = templates[question_type]
                question = f"Is this sentence correct? {statement} Answer yes or no."
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
    for seed in [42, 43, 44]:  # Run with different seeds for robustness
        all_results[seed] = {} 
        print(f"\nRunning with seed: {seed}")
        for model_name in ["llava", "internvl", "paligemma"]:
            print(f"Evaluating model: {model_name}")
            # use multiprocessing to avoid cuda crash and collect all results
            with mp.Pool(processes=1) as pool:
                results = pool.apply(get_4_object_accuracy, args=(model_name, "4_shapes_same_dataset_1000", 42))
            print(f"Results for {model_name}:")
            for question_type, result in results.items():
                accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                print(f"{question_type.capitalize()} Accuracy: {accuracy:.2f}%")
                print(f"Examples where the model got it wrong for {question_type}:")
                for example in result["examples"][:5]:  # print first 5 examples
                    print(f"Model: {model_name}, Image: {example[0]}, Question: {example[1]}, Model Answer: {example[2]}")
            all_results[seed][model_name] = results

    # save results to a json file
    with open("4_object_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # print out results in a nice format
    for seed, models in all_results.items():
        print(f"\nSeed: {seed}")
        for model_name, results in models.items():
            print(f"Model: {model_name}")
            for question_type, result in results.items():
                accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                print(f"{question_type.capitalize()} Accuracy: {accuracy:.2f}%")
                print(f"Examples where the model got it wrong for {question_type}:")
                for example in result["examples"][:5]:  # print first 5 examples
                    print(f"Model: {model_name}, Image: {example[0]}, Question: {example[1]}, Model Answer: {example[2]}")

    # show results in a table with pandas, report average and standard deviation across seeds for each model and question type
    # report just average across seeds for each model and question type
    import numpy as np
    import pandas as pd

    summary_data = []

    # Using the keys from your results dict to avoid hard-coding seeds
    available_seeds = list(all_results.keys())
    model_names = ["llava", "internvl", "paligemma"]

    print("\n--- Detailed Accuracy per Seed ---")

    for model_name in model_names:
        for question_type in question_types:
            accuracies = []
            
            # 1. Collect and Print individual seed results
            print(f"\nModel: {model_name} | Type: {question_type}")
            for seed in available_seeds:
                result = all_results[seed][model_name][question_type]
                accuracy = (result["correct"] / result["total"] * 100) if result["total"] > 0 else 0
                accuracies.append(accuracy)
                print(f"  Seed {seed}: {accuracy:.2f}%")
            
            # 2. Calculate Stats
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            # 3. Print Stats immediately for quick reading
            print(f"  >> Average: {avg_accuracy:.2f}% (±{std_accuracy:.2f})")
            
            # 4. Store for DataFrame
            summary_data.append({
                "Model": model_name,
                "Question Type": question_type,
                "Average Accuracy": avg_accuracy,
                "Std Accuracy": std_accuracy
            })

    # Create the summary table
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*30)
    print("FINAL SUMMARY TABLE")
    print("="*30)
    print(summary_df.to_string(index=False))

    # Export
    summary_df.to_csv("4_object_results_summary.csv", index=False)

    grouped_summary_data = []
    for model_name in ["llava", "internvl", "paligemma"]:
        for group in ["positive", "negative"]:
            accuracies = []
            for seed in [42, 43, 44]:
                if group == "positive":
                    result_pos = all_results[seed][model_name]["positive"]
                    result_pos_false = all_results[seed][model_name]["positive_false"]
                    accuracy_pos = (result_pos["correct"] / result_pos["total"] * 100) if result_pos["total"] > 0 else 0
                    accuracy_pos_false = (result_pos_false["correct"] / result_pos_false["total"] * 100) if result_pos_false["total"] > 0 else 0
                    accuracies.append((accuracy_pos + accuracy_pos_false) / 2)
                else:
                    result_neg = all_results[seed][model_name]["negative"]
                    result_neg_false = all_results[seed][model_name]["negative_false"]
                    accuracy_neg = (result_neg["correct"] / result_neg["total"] * 100) if result_neg["total"] > 0 else 0
                    accuracy_neg_false = (result_neg_false["correct"] / result_neg_false["total"] * 100) if result_neg_false["total"] > 0 else 0
                    accuracies.append((accuracy_neg + accuracy_neg_false) / 2)
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            grouped_summary_data.append({
                "Model": model_name,
                "Question Group": group,
                "Average Accuracy": avg_accuracy,
                "Std Accuracy": std_accuracy
            })
    grouped_summary_df = pd.DataFrame(grouped_summary_data)
    print("\nGrouped Summary of Results:")
    print(grouped_summary_df)
    grouped_summary_df.to_csv("4_shapes_grouped_summary.csv")


if __name__ == "__main__":
    # run_whatsup_results()
    # run_2_object_results()
    # run_4_object_results()
    get_2_object_accuracy("llava", "whatsup", 42)