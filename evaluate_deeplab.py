"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import collections
import glob
import math
import time

import tensorflow as tf
import numpy as np
import scipy as scp

# sys.path.append("/Users/i859032/src/github.com/TgithubJ/tensorflow-deeplab-resnet/")
from deeplab_resnet import DeepLabResNetModel, prepare_label



IMG_RGB_MEAN = np.array([71.61794271, 78.70845457, 56.54645427], dtype=np.float32)
IMG_RGB_STD = np.array([29.02917546, 26.41235474, 22.43456377], dtype=np.float32)

# DATA_DIRECTORY = '/Users/i859032/images/Red_Bull/512X512/data/val'
DATA_DIRECTORY = '../../../data/redbull/val'
IGNORE_LABEL = 255
NUM_CLASSES = 2
NUM_STEPS = 5552 # Number of images in the validation set.
# RESTORE_FROM = '/Users/i859032/Desktop/jupyter/Image_Segmentation/Results/redbull/20170626'
RESTORE_FROM = '../../../data/redbull_ckpt/20170626_best'
# RESTORE_FROM = 'cour'

LOG_TO = RESTORE_FROM + '/log'
Examples = collections.namedtuple("Examples", "inputs, targets, count")


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--is-training", action="store_false",
                        help="Whether to updates the running means and variances during the training.")    
    return parser.parse_args()


def load_examples(args, Mean_Subtract=True, RGB_to_BGR=True):
    if args.data_dir is None or not os.path.exists(args.data_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(args.data_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(args.data_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=args.is_training)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])
        width = tf.shape(raw_input)[1] # [height, width, channels]
        
        img = (tf.cast(raw_input[:,:width//2,:], dtype=tf.float32) - IMG_RGB_MEAN)/IMG_RGB_STD

        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        a_images = tf.concat(axis=2, values=[img_b, img_g, img_r])

        b_images = raw_input[:,width//2:,:]
        inputs, targets = [a_images, b_images]

    input_images = inputs
    targets1 = tf.squeeze(tf.to_int32(targets[:,:,2:]), squeeze_dims=[2])
    targets_one_hot = tf.one_hot(targets1, depth=args.num_classes, axis=-1)
    # targets2 = transform(targets_one_hot)
    target_images = tf.argmax(targets_one_hot, axis=-1)

    # print(input_images.get_shape())
    # print(target_images.get_shape())


    # paths_batch, inputs_batch, targets_batch = tf.train.batch(
    #     [paths, input_images, target_images], 
    #     batch_size=1)

    return Examples(
        inputs=input_images,
        targets=target_images,
        count=len(input_paths),
    )


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    
    if not os.path.exists(LOG_TO):
        os.makedirs(LOG_TO)

    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        examples = load_examples(args)
        print("examples count = %d" % examples.count)

        image_batch, label_batch = tf.expand_dims(examples.inputs, dim=0), tf.expand_dims(examples.targets, dim=0)
        h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
        image_batch075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
        image_batch05 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.5)), tf.to_int32(tf.multiply(w_orig, 0.5))]))
    

    # Create network.
    with tf.variable_scope('', reuse=False):
        net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)
    with tf.variable_scope('', reuse=True):
        net075 = DeepLabResNetModel({'data': image_batch075}, is_training=False, num_classes=args.num_classes)
    with tf.variable_scope('', reuse=True):
        net05 = DeepLabResNetModel({'data': image_batch05}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output100 = net.layers['fc1_voc12']
    raw_output075 = tf.image.resize_images(net075.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    raw_output05 = tf.image.resize_images(net05.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    
    raw_output = tf.reduce_max(tf.stack([raw_output100, raw_output075, raw_output05]), axis=0)
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.



    # Which variables to load.
    # restore_var = tf.global_variables()
    
    # Predictions.
    # raw_output = net.layers['fc1_voc12']
    # raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    # raw_output = tf.argmax(raw_output, dimension=3)
    # pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    # mIoU
    pred = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    # weights = tf.cast(tf.less_equal(gt, args.num_classes - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    # mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes, weights=weights)
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes)
    

    loader = tf.train.Saver()

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
    
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(tf.local_variables_initializer())
    
        # Load weights.
        checkpoint_file = tf.train.latest_checkpoint(args.restore_from)
        loader.restore(sess, checkpoint_file)
        print("Restored model parameters from {}".format(checkpoint_file))
        # loader = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        # loader.restore(sess, checkpoint_file) 
        # loader = tf.train.Saver(var_list=restore_var)
        
        # if args.restore_from is not None:
        #     load(loader, sess, checkpoint_file)
        

        # Create queue coordinator.cd
        coord = tf.train.Coordinator()

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        
        # Iterate over training steps.
        # iou_list = []
        start_time = time.time()
        for step in range(args.num_steps):
            output, _ , _ = sess.run([raw_output, pred, update_op])
            if step % 20 == 0:
                print('step {:d}'.format(step))

            # print(preds.shape)
            # real_output = np.squeeze(preds[0], axis=-1)
            # print(real_output.shape)

            # reshaped_pred = np.reshape(output, (512, 512))
            # print(output.shape)
            # scp.misc.imsave(LOG_TO+'/eval_output_'+str(step)+'.png', output[0])
        
            # temp_iou = sess.run(mIoU)
            # iou_list.append(float(temp_iou))


        duration = time.time() - start_time
        print('{:.3f} Time Consumed for {} pictures'.format(duration, examples.count))
        # print(iou_list)
        print('Mean IoU: {:.3f}'.format(sess.run(mIoU)))
        coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    main()
