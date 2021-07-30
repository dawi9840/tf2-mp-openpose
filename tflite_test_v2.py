import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import tensorflow as tf
import matplotlib.pylab as plt
import cv2
import numpy as np

tflite_model_file = 'tflite/0620_epoch300/openpose_singlenet.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)

inp_details = interpreter.get_input_details()
out_details = sorted(interpreter.get_output_details(), key=lambda k: k['index']) 

inp_index = interpreter.get_input_details()[0]["index"]
out_index = interpreter.get_output_details()[0]["index"]
paf_idx = out_details[-2]["index"]
heatmap_idx = out_details[-1]["index"]

# M1: tf.print()
print("[tflite details]:\n--------------------------------------------------")
tf.print(type(inp_details), ", inp_details:\n\n", inp_details, end="\n--------------------------------------------------\n")
tf.print(type(out_details), ", out_details:\n", out_details)

# M2: print()
'''
print(type(inp_details), ", inp_details:\n", inp_details, end="\n--------------------------------------------------\n")
print(type(out_details), ", out_details:\n")
for i in range(0, len(out_details)):
    print(out_details[i], end="\n--------------------------------------------------\n")
'''

print("\n[index]:")
print("inp_index:", inp_index)
print("out_index:", out_index)
print("paf_idx:", paf_idx)
print("heatmap_idx:", heatmap_idx)

#=====================================#
# Load sample image and run the model #
#=====================================#

test_image = 'resources/ski_224.jpg'

img = cv2.imread(test_image) # B,G,R order
img = np.expand_dims(img, 0)

'''
input_tensor= tf.convert_to_tensor(img, np.uint8)

interpreter.allocate_tensors()

interpreter.set_tensor(inp_index, input_tensor)

interpreter.invoke()

heatmaps = interpreter.get_tensor(heatmap_idx)
pafs = interpreter.get_tensor(paf_idx)

'''