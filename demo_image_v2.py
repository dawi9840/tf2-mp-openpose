import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import cv2
import numpy as np
import importlib

from estimation.config import get_default_configuration
from estimation.coordinates import get_coordinates
from estimation.connections import get_connections
from estimation.estimators import estimate
from estimation.renderers import draw

from train_singlenet_mobilenetv3 import register_tf_netbuilder_extensions

image = "resources/ski_224.jpg"
output_image = "demo2.png"
create_model_fn = "create_openpose_singlenet"
weights_path = "output_singlenet/0620_epoch300/openpose_singlenet"
paf_idx = 2
heatmap_idx = 3

if __name__ == '__main__':
    register_tf_netbuilder_extensions()

    module = importlib.import_module('models')
    create_model = getattr(module, create_model_fn)

    model = create_model()
    model.load_weights(weights_path)

    # B,G,R order
    img = cv2.imread(image)  
    input_img = img[np.newaxis, :, :, [2, 1, 0]]
    inputs = tf.convert_to_tensor(input_img)

    outputs = model.predict(inputs)
    pafs = outputs[paf_idx][0, ...]
    heatmaps = outputs[heatmap_idx][0, ...]

    cfg = get_default_configuration()
    coordinates = get_coordinates(cfg, heatmaps)
    connections = get_connections(cfg, coordinates, pafs)
    skeletons = estimate(cfg, connections)
    output = draw(cfg, img, coordinates, skeletons, resize_fac=8)
    cv2.imwrite(output_image, output)
    print(f"Output saved: {output_image}")