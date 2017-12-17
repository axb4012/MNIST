from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
#import pdb
import glob
import cPickle

import numpy as np
import tensorflow as tf


def readFromTFRecords(filename, batch_size, num_epochs, img_shape, num_threads=2, min_after_dequeue=1000):
    """
    Args:
        filename: the .tfrecords file we are going to load
        batch_size: batch size
        num_epoch: number of epochs, 0 means train forever
        img_shape: image shape: [height, width, channels]
        num_threads: number of threads
        min_after_dequeue: defines how big a buffer we will randomly sample from,
            bigger means better shuffling but slower start up and more memory used.
            (capacity is usually min_after_dequeue + (num_threads + eta) * batch_size)

    Return:
        images: (batch_size, height, width, channels)
        labels: (batch_size)
    """

    def read_and_decode(filename_queue, img_shape):
        """Return a single example for queue"""
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            }
        )
        # some essential steps
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, img_shape)  # THIS IS IMPORTANT
        image.set_shape(img_shape)
        image = tf.cast(image, tf.float32) * (1)  # set to [0, 1]

        sparse_label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(sparse_label, 151)

        return image, label

    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

    image, sparse_label = read_and_decode(filename_queue, img_shape)  # share filename_queue with multiple threads

    # tf.train.shuffle_batch internally uses a RandomShuffleQueue
    images, sparse_labels = tf.train.shuffle_batch(
        [image, sparse_label], batch_size=batch_size, num_threads=num_threads,
        min_after_dequeue=min_after_dequeue,
        capacity=min_after_dequeue + (num_threads + 1) * batch_size
    )


    return images, sparse_labels


class Queue_loader():
    # This queue loader use cifar10 as example data
    def __init__(self, batch_size, num_epochs, num_threads=2, min_after_dequeue=500, train=True):
        if train:
            filename = './tf_rec_for_plate/plate_tfrec'
        else:
            filename = 'test_package.tfrecords'



        img_shape = [224, 224, 3]
        self.num_examples = 26000 if train else 1000
        # the above 2 lines are set manually assuming we have already generated .tfrecords file
        self.num_batches = int(self.num_examples / batch_size)

        # Second, we are going to read from .tfrecords file, this contains several steps
        self.images, self.labels = readFromTFRecords(filename, batch_size, num_epochs,
                                                     img_shape, num_threads, min_after_dequeue)

# done


