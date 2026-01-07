import json
from plotter2 import Plotter
from PIL import Image

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

    return processor, model

with open('whatsup/controlled_images/detections.json', 'r') as file:
    data = json.load(file)

# print out first 5 entries
for img_path, detections in list(data.items())[0]:
    print(f"Image Path: {img_path}")
    #print(f"Detections: {detections}")
    for det in detections:
        print(det['name'], det['bbox'])

    processor, model = load_model("llava")
    full_img_path = f"whatsup/controlled_images/{img_path}"
    plotter = Plotter(full_img_path)
    plotter.set_model(model, processor)
    left_shape = plotter.get_left_shapes()
    
    # print size of image
    image = Image.open(full_img_path)
    width, height = image.size
    print(f"Image size: {width}x{height}")