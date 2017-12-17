import tensorflow as tf
import numpy as np

class CNN:

    def __init__(self, lr, batch_size, num_batches):
        self.lr = lr
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.g_step = tf.contrib.framework.get_or_create_global_step()

    def conv_layer(self, name, image, f_size, in_c, out_c, strides=1):
        with tf.variable_scope(name):
            f, b = self.get_conv_var(f_size, in_c, out_c, name)
            conv = tf.nn.conv2d(image, f, [1, strides, strides, 1], padding='SAME')
            return tf.nn.bias_add(conv, b)

    def activation(self, conv, activation_type='RELU'):
        return tf.nn.relu(conv)

    def get_conv_var(self, f_size, in_c, out_c, name):
        f = tf.get_variable(name + '_f', [f_size, f_size, in_c, out_c], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name + '_b', [out_c], initializer=tf.truncated_normal_initializer())
        return f, b

    def max_pool(self, conv, name, k = 2, s = 2):
        return tf.nn.max_pool(conv, [1, k ,k , 1], [1, s, s, 1], padding='VALID', name=name)

    def fc_layer(self, bottom, in_c, out_c, name):
        with tf.variable_scope(name):
            w, b = self.get_fc_var(in_c, out_c, name)
            #self.i1 = w
            x = bottom
            #x = tf.reshape(bottom, [-1, in_c])
            #if name == 'fc3':
            #    self.i1 = x
            #    self.i2 = bottom
            return tf.nn.xw_plus_b(x, w, b)

    def get_fc_var(self, in_c, out_c, name):
        w = tf.get_variable(name + '_w', [in_c, out_c], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name + '_b', [out_c], initializer=tf.contrib.layers.xavier_initializer())
        return w, b

    def lrn_layer(self, bottom, name, bias=1.0, alpha=1e-3 / 9.0, beta=0.75):
        return tf.nn.local_response_normalization(bottom, depth_radius=4, bias=bias, alpha=alpha, beta=beta, name=name)

    def optimizer(self, lr):
        optim = tf.train.AdagradOptimizer(lr,name='Adam')
        return optim
        #return tf.train.GradientDescentOptimizer(*args)

    def build_CNN(self, images):
        #images = tf.reduce_mean(images,axis=3,name='Exapnd_Axis')
        #images = tf.expand_dims(images, axis=3)
        #self.i1 = images[1]
        conv = self.conv_layer('conv1', images, 3, 3, 64, 1)
        activ = self.activation(conv)
        conv = self.conv_layer('conv1_2', activ, 3, 64, 64, 1)
        activ = self.activation(conv)
        pool =  self.max_pool(activ, 'pool1')
        net = tf.nn.batch_normalization(x=pool,mean=0,variance=1,variance_epsilon=0.001,name='batch_norm1',offset=None,scale=None)
        #net = self.lrn_layer(pool, 'lrn1')

        conv = self.conv_layer('conv2', net, 3, 64, 128, 1)
        activ = self.activation(conv)
        conv = self.conv_layer('conv2_2', activ, 3, 128, 128, 1)
        activ = self.activation(conv)
        pool =  self.max_pool(activ, 'pool2')
        net = tf.nn.batch_normalization(x=pool,mean=0,variance=1,variance_epsilon=0.001,name='batch_norm2',offset=None,scale=None)


        conv = self.conv_layer('conv3', net, 3, 128, 256, 1)
        activ = self.activation(conv)
        conv = self.conv_layer('conv3_2', activ, 3, 256, 256, 1)
        activ = self.activation(conv)

        conv = self.conv_layer('conv3_3', activ, 3, 256, 256, 1)
        activ = self.activation(conv)
        conv = self.conv_layer('conv3_4', activ, 3, 256, 256, 1)
        activ = self.activation(conv)

        pool =  self.max_pool(activ, 'pool3')
        #self.i1 = pool
        net = tf.nn.batch_normalization(x=pool,mean=0,variance=1,variance_epsilon=0.001,name='batch_norm3',offset=None,scale=None)
        #self.i1 = net
        #net = tf.nn.batch_normalization()
        #net = self.lrn_layer(pool, 'lrn2')

        conv = self.conv_layer('conv4', net, 3, 256, 512, 1)
        activ = self.activation(conv)
        conv = self.conv_layer('conv4_2', activ, 3, 512, 512, 1)
        activ = self.activation(conv)

        conv = self.conv_layer('conv4_3', activ, 3, 512, 512, 1)
        activ = self.activation(conv)
        conv = self.conv_layer('conv4_4', activ, 3, 512, 512, 1)
        activ = self.activation(conv)

        pool =  self.max_pool(activ, 'pool4')
        #self.i1 = pool
        net = tf.nn.batch_normalization(x=pool,mean=0,variance=1,variance_epsilon=0.001,name='batch_norm4',offset=None,scale=None)


        conv = self.conv_layer('conv5', net, 3, 512, 512, 1)
        activ = self.activation(conv)
        conv = self.conv_layer('conv5_2', activ, 3, 512, 512, 1)
        activ = self.activation(conv)

        conv = self.conv_layer('conv5_3', activ, 3, 512, 512, 1)
        activ = self.activation(conv)
        conv = self.conv_layer('conv5_4', activ, 3, 512, 512, 1)
        activ = self.activation(conv)

        pool =  self.max_pool(activ, 'pool5')
        #self.i1 = pool
        net = tf.nn.batch_normalization(x=pool,mean=0,variance=1,variance_epsilon=0.001,name='batch_norm5',offset=None,scale=None)

        #conv = self.conv_layer('conv6', net, 3, 512, 1024, 1)
        #activ = self.activation(conv)
        #pool =  self.max_pool(activ, 'pool6')
        #self.i1 = pool
        #net = tf.nn.batch_normalization(x=pool,mean=0,variance=1,variance_epsilon=0.001,name='batch_norm6',offset=None,scale=None)

        dim = np.prod(net.shape[1:]).value
        #self.i1 = tf.reshape(net, [-1, dim])
        net = tf.nn.relu(self.fc_layer(tf.reshape(net, [-1, dim]), dim, 4096, name='fc3'))
        #self.i1 = net
        net = tf.nn.relu(self.fc_layer(net, 4096, 4096, name='fc4'))

        self.logits = self.fc_layer(net, 4096, 151, name='logits')

    def predictions(self, cross_entropy_loss, labels):
        preds = tf.argmax(cross_entropy_loss, axis=1)
        self.t = preds
        normal_labels = tf.argmax(labels,axis=1)
        self.labl = normal_labels
        _,accuracy = tf.metrics.accuracy(preds,normal_labels)
        return accuracy

    def loss(self, labels):
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)
        #self.i1 = cross_entropy_loss
        y_pred = tf.nn.softmax(self.logits)
        #self.i1 = y_pred
        #print(y_pred)
        self.accuracy = self.predictions(y_pred, labels)
        self.loss_op = tf.reduce_mean(cross_entropy_loss)
        self.cross_e_loss = self.t, self.labl

    def train(self,acc=0,c_e = 0):
        self.lr = tf.train.exponential_decay(self.lr, self.g_step, self.batch_size * self.num_batches, 0.9, staircase=True)
        #self.lr = 1e-4
        if acc==0 and c_e == 0:
            #ret_value = self.optimizer()
            ret_value =  self.optimizer(self.lr).minimize(self.loss_op, global_step=self.g_step)
        else:
            if acc==1 and c_e == 0:
                ret_value = self.accuracy
            else:
                ret_value = self.cross_e_loss
        return ret_value
    