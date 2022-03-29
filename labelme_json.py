import json
import numpy as np
import rasterio
import rasterio.features
from tifffile import imsave


def json_annotation_to_array(image_path, json_path, masked=True):
    """Transforms the json annotations in the JSON file at 
       'json_path' of a Geotiff at 'image_path') annotated with
       LabelMe to a numpy array (optionally masked)."""  # Open the original Geotiff and save the dimensions and optional
    # mask data
    with rasterio.open(image_path) as gtif:
        height = gtif.height
        width = gtif.width
        if masked == True:
            data = gtif.read(masked=True)
            data_mask = data.mask

    # Open the JSON file
    with open(json_path, 'rb') as json_file:
        labelme_json = json.load(json_file)

    # Create a list of "geometry-value" pairs for use with the
    # rasterize function from rasterio
    shapes = []
    for shape in labelme_json['shapes']:
        coord_list = []
        for point in shape['points']:
            coord_list.append((point[0], point[1]))
        shapes.append(({'coordinates': [coord_list],
                        'type': 'Polygon'}, 255))

    # Create the array
    labelme_array = rasterio.features.rasterize([(g, 255) for g, v
                                                 in shapes], out_shape=(height, width),
                                                all_touched=True)

    # Apply optional mask
    if masked == True:
        labelme_array = np.ma.array(labelme_array)
        labelme_array.mask = data_mask

    return labelme_array


image_path = '/home/ben/Documents/landsat_downloads/scene4/landsat_bands/LC08_L1TP_045033_20200901_20200906_02_T1_B4.TIF'
json_path = '/home/ben/Documents/landsat_downloads/scene4/labels_json/labels.json'

xd = json_annotation_to_array(image_path, json_path, False)
imsave('/home/ben/Documents/landsat_downloads/scene4/labels_json/labels.tif', xd)
print("xd")
