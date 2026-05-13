# **Probing Negation Undertanding in VLMs With Object Attention**


Create environment
```
conda env create -f environment.yml
```
### Datasets

Unzip each of the three datasets 
```
unzip datasets/binary.zip -d .
unzip datasets/multary.zip -d .
unzip datasets/whatsup.zip -d .
```
### Visualise Bounding Box

View the bounding box dimensions one the image and attention map
```
python scripts/visualise_bbox.py --image_path "binary/images/image_0001.png"
```
Saves to bbox_map.png and bbox_image.png

### Generate Data 

Generate data for bar chart figures and layer-wise plots. Use the image dir flag to choose which of the three datasets to generate for all three models. 
```
python scripts/generate_data.py --image_dir "binary"
```
To generate the layer-wise graphs for left vs right side mentioned
```
 python scripts/generate_data.py --image_dir "binary"--data_type "leftright"
 ```
 Data will be saved in a folder called `data_saves`

 ### Plot graphs
To plot:
 - bar graph
 - layer-wise plots for binary, multary and whatsup datasets (one figure for the three models)
 - left vs right plots for the binary dataset
 run the following script after generating the data. Data will be plotted using values in the `data_saves` folder
 ```
 python scripts/plot_graphs.py
```

### Downstream
Use
```
python scripts/downstream.py --function "gen"
```
to generate json files with downstream results in. These will be saved to `2_object_results.json`, `4_object_results.json` and `whatsup_results.json`.

To print the results of the downstream task use
```
python scripts/downstream.py
```