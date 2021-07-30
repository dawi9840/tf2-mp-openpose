from math import modf
import tensorflow as tf
import cv2
import numpy as np
import importlib
import time

from estimation.config import get_default_configuration
from estimation.coordinates import get_coordinates
from estimation.connections import get_connections
from estimation.estimators import estimate
from estimation.renderers import draw

from train_singlenet_mobilenetv3 import register_tf_netbuilder_extensions


video_path = "resources/sample3.mp4"    
output_video = "sample3_out2.mp4"
create_model_fn = "create_openpose_singlenet"
weights_path = "output_singlenet/0629_epoch300/openpose_singlenet"

def process_frame(cropped, heatmap_idx, model, paf_idx, output_resize_factor):

    input_img = cropped[np.newaxis, ...]
    inputs = tf.convert_to_tensor(input_img)
    outputs = model.predict(inputs)
    pafs = outputs[paf_idx][0, ...]
    heatmaps = outputs[heatmap_idx][0, ...]

    cfg = get_default_configuration()
    coordinates = get_coordinates(cfg, heatmaps)
    connections = get_connections(cfg, coordinates, pafs)
    skeletons = estimate(cfg, connections)
   
    return draw(cfg, cropped, coordinates, skeletons, resize_fac=output_resize_factor)

def main(video=video_path, output_video=output_video, create_model_fn=create_model_fn, input_size=224, output_resize_factor=8, paf_idx=2, heatmap_idx=3, frames_to_analyze=None, frame_ratio=1):

    register_tf_netbuilder_extensions()

    module = importlib.import_module('models')
    create_model = getattr(module, create_model_fn)
    model = create_model(pretrained=False)
    model.load_weights(weights_path)   

    # Video reader
    cam = cv2.VideoCapture(video)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if frames_to_analyze is None:
        frames_to_analyze = video_length

    # Video writer
    output_fps = input_fps / frame_ratio
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = orig_image.shape[1]
    h = orig_image.shape[0]
    scale = input_size / w if w < h else input_size / h
    out = cv2.VideoWriter(output_video, fourcc, output_fps, (input_size, input_size))

    i = 1
    sum_fps = 0

    while (cam.isOpened()) and ret_val is True and i < frames_to_analyze:
        if i % frame_ratio == 0:
            # print("orig_fps : {0}".format(cam.get(cv2.CAP_PROP_FPS)))
            tic = time.time()

            im = cv2.resize(orig_image, (0, 0), fx=scale, fy=scale)
            new_w = im.shape[1]
            new_h = im.shape[0]
            if new_w > new_h:
                offset = (new_w - input_size) // 2
                cropped = im[0: input_size, offset: offset + input_size]
            else:
                offset = (new_h - input_size) // 2
                cropped = im[offset: offset + input_size, 0: input_size]

            canvas = process_frame(cropped, heatmap_idx, model, paf_idx, output_resize_factor)

            # print('Processing frame: ', i)
            toc = time.time()
            # print('processing time is %.5f' % (toc - tic))
            sum_fps += 1 / (toc - tic)
            # print("fps: {}".format(1 / (toc - tic)))
            print("avage fps:{}, frame: {}, fps: {}".format((sum_fps/i), i, (1 / (toc - tic))))
            
            out.write(canvas)

        ret_val, orig_image = cam.read()
        
        i += 1
    print("Done!")

def main_cap(video=0, input_size=224, output_resize_factor=8, paf_idx=2, heatmap_idx=3, frame_ratio=1):

    register_tf_netbuilder_extensions()

    module = importlib.import_module('models')
    create_model = getattr(module, create_model_fn)
    model = create_model(pretrained=False)
    model.load_weights(weights_path)

    cam = cv2.VideoCapture(video)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    ret_val, orig_frame = cam.read()
    w = orig_frame.shape[1]
    h = orig_frame.shape[0]
    scale = input_size / w if w < h else input_size / h
    # w: 640, h:480, scale:0.4666666666666667
    # print("w: {}, h:{}, scale:{}".format(w, h, scale))

    i = 1
    sum_fps = 0

    while (cam.isOpened()) and ret_val is True:
        if i % frame_ratio == 0:

            tic = time.time()

            im = cv2.resize(orig_frame, (0, 0), fx=scale, fy=scale)
            new_w = im.shape[1]
            new_h = im.shape[0]
            if new_w > new_h:
                offset = (new_w - input_size) // 2
                cropped = im[0: input_size, offset: offset + input_size]
            else:
                offset = (new_h - input_size) // 2
                cropped = im[offset: offset + input_size, 0: input_size]

            cropped = process_frame(cropped, heatmap_idx, model, paf_idx, output_resize_factor)

            print('Processing frame: ', i)
            toc = time.time()
            print('processing time is %.5f' % (toc - tic))
            sum_fps += 1 / (toc - tic)
            # print("fps: {}".format(1 / (toc - tic)))
            print("avage fps:{}, frame: {}, fps: {}".format((sum_fps/i), i, (1 / (toc - tic))))

        ret_val, orig_frame = cam.read()

        i += 1

        cv2.imshow('frame', cropped)

        if cv2.waitKey(1) == ord('q'):
            break


    cam.release()
    cv2.destroyAllWindows()
    print("Done!")

def trun_on_camera(video=0, is_resize=False):
    if is_resize == False:
        cam = cv2.VideoCapture(video)
        if not cam.isOpened():
            print("Cannot open camera")
            exit()

        while(True):
            # Capture frame-by-frame
            ret, frame = cam.read()
            # w = frame.shape[1]
            # h = frame.shape[0]
            # scale = 224 / w if w < h else 224 / h
            # print("w: {}, h:{}, scale:{}".format(w, h, scale))

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
    else:
        print("resize")
        input_size = 224
        cam = cv2.VideoCapture(video)

        ret_val, orig_frame = cam.read()
        w = orig_frame.shape[1]
        h = orig_frame.shape[0]
        scale = input_size / w if w < h else input_size / h
        # w: 640, h:480, scale:0.4666666666666667
        # print("w: {}, h:{}, scale:{}".format(w, h, scale))

        i = 0

        while (cam.isOpened()) and ret_val is True:
            tic = time.time()
            im = cv2.resize(orig_frame, (0, 0), fx=scale, fy=scale)
            new_w = im.shape[1]
            new_h = im.shape[0]

            if new_w > new_h:
                offset = (new_w - input_size) // 2
                cropped = im[0: input_size, offset: offset + input_size]
            else:
                offset = (new_h - input_size) // 2
                cropped = im[offset: offset + input_size, 0: input_size]       

            ret_val, orig_frame = cam.read()
            toc = time.time()
            print('fps: {}'.format(1 / (toc - tic)))
            # print('Processing frame: ', i)
            # print("cropped_w: {}, cropped_h: {}".format(cropped.shape[1], cropped.shape[0]))

            i += 1

            cv2.imshow('frame', cropped)
            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # trun_on_camera(video=1, is_resize=True)
    # main_cap(video=0)