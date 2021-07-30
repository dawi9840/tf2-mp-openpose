import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from models import create_openpose_singlenet
from models import create_openpose_2branches_vgg
from tf_netbuilder_ext.extensions import register_tf_netbuilder_extensions

output_weights = 'output_singlenet/0629_epoch300/openpose_singlenet'
output_weights2 = 'output_2br_vgg/0723_epoch100/openpose_2br_vgg'

if __name__ == '__main__':
    
    # Load model and weights: open_pose_single_net.
    model = create_openpose_singlenet(pretrained=False)
    model.load_weights(output_weights)
    print(model.summary())

    import sys
    sys.exit()
    # Load model and weights: OpenPose2BrVGG.
    register_tf_netbuilder_extensions()
    model2 = create_openpose_2branches_vgg(pretrained=True, training=False)
    # model2.load_weights(output_weights2)
    print(model2.summary())