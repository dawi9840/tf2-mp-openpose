# tf2-mp-openpose
Tensorflow2.x_Realtime_Multi-Person_Openpose_Estimation 

This repo using Tensorflow 2.0 Realtime Multi-Person Pose Estimation which clone from [tensorflow_Realtime_Multi-Person_Pose_Estimation ](https://github.com/michalfaber/tensorflow_Realtime_Multi-Person_Pose_Estimation). 
I study it and modify some code that I can understand clearly. 

# Scripts and notebooks

This project contains the following scripts and jupyter notebooks:

**train_singlenet_mobilenetv3.py** - training code for the openpose_singlenet model presented in this paper [Single-Network Whole-Body Pose Estimation](https://arxiv.org/abs/1909.13423).

**train_2br_vgg_v2.py and train_2br_vgg_v2.ipynb** - training code for the old CMU model (2017). This is a new version of the training code from the old repo [keras_Realtime_Multi-Person_Pose_Estimation](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation). It has been upgraded to Tensorflow 2.0.

**convert_to_tflite.py** - conversion of trained models into *TFLite*.

**demo_image.py and demo_image_v2.py** - pose estimation on the provided image.

**demo_video_v2.py and demo_video_v2.py.ipynb** - modify the  demo_video.py to pose estimation on the provided video.

**inspect_dataset.ipynb** - helper notebook to get more insights into what is generated from the datasets.

**test_openpose_singlenet_model.ipynb** - helper notebook to preview the predictions from the singlenet model.

**test_openpose_2br_vgg_v2.ipynb** - helper notebook to preview the predictions from the original vgg-based model.

**test_tflite_v2.ipynb** - helper notebook to verify exported *TFLite* model.
  

# Installation

## Prerequisites

* download [dataset and annotations](http://cocodataset.org/#download) into a separate folder datasets, outside of this project:
```bash
    ├── datasets
    │   └── coco_2017_dataset
    │       ├── annotations
    │       │   ├── person_keypoints_train2017.json
    │       │   └── person_keypoints_val2017.json
    │       ├── train2017/*
    │       └── val2017/*
    └── tensorflow_Realtime_Multi-Person_Pose_Estimation/*
```
                
* RTX 30 series GPU.
* Package need tensorflow-gpu above version 2.2.


## Install

**Virtualenv**

I use conda env to buil the env.
```bash
conda create --name mp-pose python=3.7
conda activate mp-pose
pip install -r requirements.txt

```

## Examples
```bash
python convert_to_tflite.py --weights=[path to saved weights] --tflite-path=openpose_singlenet.tflite --create-model-fn=create_openpose_singlenet
python demo_image.py --image=resources/ski_224.jpg --output-image=out1.png --create-model-fn=create_openpose_singlenet
python demo_image.py --image=resources/ski_368.jpg --output-image=out2.png --create-model-fn=create_openpose_2branches_vgg
python demo_video.py --video=resources/sample1.mp4 --output-video=sample1_out1.mp4 --create-model-fn=create_openpose_2branches_vgg --input-size=368 --output-resize-factor=8 --paf-idx=10 --heatmap-idx=11
python demo_video.py --video=resources/sample1.mp4 --output-video=sample1_out2.mp4 --create-model-fn=create_openpose_singlenet --input-size=224 --output-resize-factor=8 --paf-idx=2 --heatmap-idx=3
```

