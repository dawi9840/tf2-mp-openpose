import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataset.generators import get_dataset, get_dataset_with_masks

annot_path_val = '../datasets/coco_2017_dataset/annotations/person_keypoints_val2017.json'
img_dir_val = '../datasets/coco_2017_dataset/val2017/'

ds, ds_size = get_dataset(annot_path_val, img_dir_val, batch_size=10)

