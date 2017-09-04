"""Training script with multi-scale inputs for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import collections
import glob
import random
import math

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, inv_preprocess, prepare_label

# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IMG_RGB_MEAN = np.array([104.42819731, 111.76448516, 125.9762497], dtype=np.float32)
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")

BATCH_SIZE = 2
DATA_DIRECTORY = '../../../data/redbull/train'
# DATA_DIRECTORY = '/Users/i859032/images/Red_Bull/512X512/data/train'
DATA_LIST_PATH = './dataset/train.txt'
GRAD_UPDATE_EVERY = 10
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 3
NUM_STEPS = 500001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './deeplab_resnet_init.ckpt'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = '../../../data/redbull_ckpt/20170622'
# SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
CROP_SIZE = 512
SCALE_SIZE = 522


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--grad-update-every", type=int, default=GRAD_UPDATE_EVERY,
                        help="Number of steps after which gradient update is applied.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to update the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')


def load_examples(args):
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
        
        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        # raw_input[:,:width//2,:] -= IMG_MEAN
        img = tf.cast(raw_input[:,:width//2,:], tf.float32) - IMG_RGB_MEAN
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        a_images = tf.concat(axis=2, values=[img_b, img_g, img_r])

        # a_images = preprocess(tf.image.convert_image_dtype(raw_input[:,:width//2,:], dtype=tf.float32))
        b_images = raw_input[:,width//2:,:]
        inputs, targets = [a_images, b_images]


    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if False:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [SCALE_SIZE, SCALE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, SCALE_SIZE - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if SCALE_SIZE > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif SCALE_SIZE < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        targets1 = tf.squeeze(tf.to_int32(targets[:,:,2:]), squeeze_dims=[2])
        targets_one_hot = tf.one_hot(targets1, depth=args.num_classes, axis=-1)
        targets2 = transform(targets_one_hot)
        target_images = tf.argmax(targets2, axis=-1)

    paths_batch, inputs_batch, targets_batch = tf.train.shuffle_batch( 
        [paths, input_images, target_images],
        batch_size=args.batch_size,
        capacity=3000,
        num_threads=2,
        min_after_dequeue=1000)
    
    steps_per_epoch = int(math.ceil(len(input_paths) / args.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    # h, w = map(int, args.input_size.split(','))
    # input_size = (h, w)
    
    tf.set_random_seed(args.random_seed)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        examples = load_examples(args)
        print("examples count = %d" % examples.count)

        final_input = tf.placeholder_with_default(examples.inputs, shape=(None, CROP_SIZE, CROP_SIZE, 3), name="final_input")
        image_batch, label_batch = [final_input, examples.targets]
        image_batch075 = tf.image.resize_images(image_batch, [int(CROP_SIZE * 0.75), int(CROP_SIZE * 0.75)])
        image_batch05 = tf.image.resize_images(image_batch, [int(CROP_SIZE * 0.5), int(CROP_SIZE * 0.5)])
    
    # Create network.
    with tf.variable_scope('', reuse=False):
        net = DeepLabResNetModel({'data': image_batch}, is_training=args.is_training, num_classes=args.num_classes)
    with tf.variable_scope('', reuse=True):
        net075 = DeepLabResNetModel({'data': image_batch075}, is_training=args.is_training, num_classes=args.num_classes)
    with tf.variable_scope('', reuse=True):
        net05 = DeepLabResNetModel({'data': image_batch05}, is_training=args.is_training, num_classes=args.num_classes)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output100 = net.layers['fc1_voc12']
    raw_output075 = net075.layers['fc1_voc12']
    raw_output05 = net05.layers['fc1_voc12']
    raw_output = tf.reduce_max(tf.stack([raw_output100,
                                         tf.image.resize_images(raw_output075, tf.shape(raw_output100)[1:3,]),
                                         tf.image.resize_images(raw_output05, tf.shape(raw_output100)[1:3,])]), axis=0)
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
    
    
    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    # raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    # raw_prediction100 = tf.reshape(raw_output100, [-1, args.num_classes])
    # raw_prediction075 = tf.reshape(raw_output075, [-1, args.num_classes])
    # raw_prediction05 = tf.reshape(raw_output05, [-1, args.num_classes])

    label_batch = tf.expand_dims(label_batch, axis=-1)
    label_proc = prepare_label(label_batch, (CROP_SIZE/8, CROP_SIZE/8), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    label_proc075 = prepare_label(label_batch, (int(CROP_SIZE/8 * 0.75), int(CROP_SIZE/8 * 0.75)), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    label_proc05 = prepare_label(label_batch, (int(CROP_SIZE/8 * 0.5), int(CROP_SIZE/8 * 0.5)), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]

    # label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    # label_proc075 = prepare_label(label_batch, tf.stack(raw_output075.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False)
    # label_proc05 = prepare_label(label_batch, tf.stack(raw_output05.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False)
    
    # raw_gt = tf.reshape(label_proc, [-1,])
    # raw_gt075 = tf.reshape(label_proc075, [-1,])
    # raw_gt05 = tf.reshape(label_proc05, [-1,])

    gt = label_proc
    gt075 = label_proc075
    gt05 = label_proc05
    
    # indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    # indices075 = tf.squeeze(tf.where(tf.less_equal(raw_gt075, args.num_classes - 1)), 1)
    # indices05 = tf.squeeze(tf.where(tf.less_equal(raw_gt05, args.num_classes - 1)), 1)
    
    # gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    # gt075 = tf.cast(tf.gather(raw_gt075, indices075), tf.int32)
    # gt05 = tf.cast(tf.gather(raw_gt05, indices05), tf.int32)
    
    # prediction = tf.gather(raw_prediction, indices)
    # prediction100 = tf.gather(raw_prediction100, indices)
    # prediction075 = tf.gather(raw_prediction075, indices075)
    # prediction05 = tf.gather(raw_prediction05, indices05)
    
    prediction = raw_output
    prediction100 = raw_output100
    prediction075 = raw_output075
    prediction05 = raw_output05

    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    loss100 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction100, labels=gt)
    loss075 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction075, labels=gt075)
    loss05 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction05, labels=gt05)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.reduce_mean(loss100) + tf.reduce_mean(loss075) + tf.reduce_mean(loss05) + tf.add_n(l2_losses)
    
    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3, name="final_output")
    
    # Image summary.
    def depreprocess(image_BGR, num_images, img_mean):
        _, h, w, c = image_BGR.shape
        
        outputs = []
        for i in range(num_images):
            input_RGB = image_BGR[i][:,:,::-1]
            input_plus_mean = input_RGB + img_mean
            input_casted = tf.cast(input_plus_mean, tf.uint8)
            outputs.append(input_casted)
        return tf.stack(outputs)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = depreprocess(image_batch, args.save_num_images, IMG_RGB_MEAN)        

    with tf.name_scope("convert_targets"):
        # targets = tf.expand_dims(label_batch, axis=-1)
        converted_targets = (label_batch * (np.round(256 / (args.num_classes-1))-1))
        converted_targets = tf.cast(converted_targets, tf.uint8)

    with tf.name_scope("convert_outputs"):
        converted_outputs = (pred * (np.round(256 / (args.num_classes-1))-1))
        converted_outputs = tf.cast(converted_outputs, tf.uint8)

    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    
    opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    # Define a variable to accumulate gradients.
    accum_grads = [tf.Variable(tf.zeros_like(v.initialized_value()),
                               trainable=False) for v in conv_trainable + fc_w_trainable + fc_b_trainable]

    # Define an operation to clear the accumulated gradients for next batch.
    zero_op = [v.assign(tf.zeros_like(v)) for v in accum_grads]

    # Compute gradients.
    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
   
    # Accumulate and normalise the gradients.
    accum_grads_op = [accum_grads[i].assign_add(grad / args.grad_update_every) for i, grad in
                       enumerate(grads)]

    grads_conv = accum_grads[:len(conv_trainable)]
    grads_fc_w = accum_grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = accum_grads[(len(conv_trainable) + len(fc_w_trainable)):]

    # Apply the gradients.
    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
    
    # summaries
    input_img_summary = tf.summary.image("inputs", converted_inputs)
    target_img_summary = tf.summary.image("targets", converted_targets)
    output_img_summary = tf.summary.image("outputs", converted_outputs)
    lr_summary = tf.summary.scalar("learning_rate", learning_rate)
    loss_summary = tf.summary.scalar("loss", reduced_loss)
    summary_op = tf.summary.merge(
        [lr_summary, loss_summary, input_img_summary, target_img_summary, output_img_summary])
    
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
    
    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }
        loss_value = 0

        # Clear the accumulated gradients.
        sess.run(zero_op, feed_dict=feed_dict)
       
        # Accumulate gradients.
        for i in range(args.grad_update_every):
            _, l_val = sess.run([accum_grads_op, reduced_loss], feed_dict=feed_dict)
            loss_value += l_val

        # Normalise the loss.
        loss_value /= args.grad_update_every

        # Apply gradients.
        if step % args.save_pred_every == 0:
            summary, _ = sess.run([summary_op, train_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            save(saver, sess, args.snapshot_dir, step)
        else:
            sess.run([train_op], feed_dict=feed_dict)

        duration = time.time() - start_time

        if step % 20 == 0:
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
