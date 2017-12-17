import tensorflow as tf

from tfrec_to_data import Queue_loader
from Run_model import CNN
#import pdb
import os
import numpy as np
import cv2
import scipy.misc as sc

tf_rec_path = './tf_rec_for_plate'

def train(b_size, ep, lr):

    queue_loader = Queue_loader(batch_size=b_size, num_epochs=ep)

    model = CNN(lr, b_size, queue_loader.num_batches)
    #pdb.set_trace()
    model.build_CNN(queue_loader.images)
    model.loss(queue_loader.labels)
    train_op = model.train()


    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    print ('Start training')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        ep = 0
        step = 1
        while not coord.should_stop():
            loss, _ = sess.run([model.loss_op, train_op])
            if step % 100 == 0:
                print ('epoch: %2d, step: %2d, loss: %.4f' % (ep + 1, step, loss))

                acc = model.train(acc=1)
                #the_things = model.train(c_e=1)

                print_val = str(sess.run(acc))
                print('Accuracy is ')
                print(print_val)
                #t = sess.run(the_things)
                #print(t)
                #if step >= 4:
                    #pdb.set_trace()

            if step % queue_loader.num_batches == 0:
                print ('epoch: %2d, step: %2d, loss: %.4f, epoch %2d done.' % (ep + 1, step, loss, ep + 1))
                checkpoint_path = os.path.join(tf_rec_path, 'cifar.ckpt')
                saver.save(sess, checkpoint_path, global_step=ep + 1)
                step = 1
                ep += 1
            else:
                step += 1
    except tf.errors.OutOfRangeError:
        print ('\nDone training, epoch limit: %d reached.' % (ep))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


    print ('Done')

train(50,10,1e-2)