from plotter import Plotter
from functions import load_model


def visualise_bbox(image_path, model_name):
    processor, model = load_model(model_name)

    plotter = Plotter(image_path)
    plotter.set_model(model, processor)

    plotter.plot_bbox_on_image(save_path="bbox_image.png")

    ## need to generate outputs for attention maps to be generated
    plotter.get_outputs("The figure is green")

    plotter.plot_bbox_on_attention_map(save_path="bbox_map.png")

if __name__ == "__main__":
    image_path = "2d_dataset_fixed_positions_1000/images/image_0000.png"
    model_name = "paligemma"
    visualise_bbox(image_path, model_name)
