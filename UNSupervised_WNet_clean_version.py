'''
Fully unsupervised version
'''
import gc
import os
import vtk
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_util
from utils import fast_batch
from utils import image_preprocessing
from utils import get_batches_fn_random
from utils import centroids_similarity_loss
from utils import tf_crop_or_pad_along_axis
from dataio import read_params


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.__version__)
vtk.vtkObject.GlobalWarningDisplayOff()

def parse_args():
    """
    Use argparse module. Santize options and return the parser.

    :return:
        args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-y','--yaml-path',help='Path to the yaml file',required=True)
    return parser.parse_args()


def name_tag(name, tag):
    '''
    Create a name tag. Return None if tag or name is None.
    This allows tensorflow to create a name if you don't provide one
    '''
    return None if tag is None or name is None else '{}_{}'.format(tag,name)


class WNetTensorflow(object):
    '''
      Fully unsupervised semantic segmentation following https://arxiv.org/pdf/1711.08506.pdf
      separable convolutions https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
    '''

    def __init__(self, params, variables):
        """
        Initializes all necessary components of the TensorFlow
        Graph.
        """
        # Assign required variables first
        self.varsM = variables

        '''
        https://github.com/mbrufau7/tfm_food_segm/blob/master/W_net_Unsupervised_%26_Centroid_Loss_1_2.ipynb
        '''
        # INITIALIZE GRAPH
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.n_input_variables = len(self.varsM)
            # Placeholders
            #self.input_images = tf.placeholder(tf.float32, shape=(None, None,None,None, self.n_input_variables),
            #                                    name='input_images')
            self.input_images = tf.placeholder(tf.float32, shape=(None, None, self.n_input_variables),name='input_images')
            self.z_voxels = tf.placeholder(tf.int32, name='z_voxels')
            self.y_voxels = tf.placeholder(tf.int32, name='y_voxels')
            self.x_voxels = tf.placeholder(tf.int32, name='x_voxels')
            self.phase = tf.placeholder(tf.bool, name='phase')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            params_z_proc = params.image_params.CROP_PAD_IMAGE_Z
            params_y_proc = params.image_params.CROP_PAD_IMAGE_Y
            params_x_proc = params.image_params.CROP_PAD_IMAGE_X
            # shape = tf.shape(self.input_images)
            self.org_x = self.x_voxels  # shape[3]
            self.org_y = self.y_voxels  # shape[2]
            self.org_z = self.z_voxels  # shape[1]

            self.input_processed = image_preprocessing(self.input_images,
                                                       params_z_proc,
                                                       params_y_proc,
                                                       params_x_proc,
                                                       self.n_input_variables,
                                                       self.org_z,
                                                       self.org_y,
                                                       self.org_x)

            # Global step - feed it in so no incrementing necessary
            # self.global_step_m = tf.placeholder(tf.int32)
            global_step = tf.Variable(0.0, trainable=False)

            def shape_so(tensor):
                # s = tensor.get_shape()
                # return tuple([s[i].value for i in range(0,len(s))])
                return tuple([d.value for d in tensor.get_shape()])

            def conv_block(inputs, filters, prev_filters, kernel, activation, phase, dil_rate, tag=None):
                # @TODO 'channels_last' is default? is this acutally a separable conv
                net_2 = tf.layers.conv3d(inputs,
                                         filters,
                                         kernel,
                                         strides=[1, 1, 1],
                                         dilation_rate=dil_rate,
                                         padding='SAME',
                                         activation=activation,
                                         kernel_initializer=keras.initializers.he_normal(),
                                         data_format='channels_last',
                                         name=name_tag('conv3d_1', tag))
                
                # net_2 = tf.layers.max_pooling3d(net_2, pool_size=kernel, strides = [1, 1, 1],
                #                                 padding = 'SAME', data_format='channels_last',
                #                                 name=name_tag('max_pooling3d', tag))
                net_2 = tf.layers.conv3d(net_2,
                                         filters,
                                         kernel,
                                         strides=[1, 1, 1],
                                         dilation_rate=[1, 1, 1],
                                         padding='SAME',
                                         activation=activation,
                                         kernel_initializer=keras.initializers.he_normal(),
                                         data_format='channels_last',
                                         name=name_tag('conv3d_2', tag))
                
                net_2 = tf.layers.max_pooling3d(net_2,
                                                pool_size=kernel,
                                                strides=[2, 2, 2],
                                                padding='SAME',
                                                data_format='channels_last',
                                                name=name_tag('max_pooling3d', tag))
                # net_2 = tf.layers.batch_normalization(net_2, center=True, scale=True, training=phase
                #                                      ,name=name_tag('batch_norm', tag)
                # net_2 = tf.layers.dropout(net_2,rate=0.20, training=phase,
                #                           name=name_tag('dropout', tag))
                return net_2

            def deconv_block(inputs,filters, prev_filters,kernel,activation, phase, dil_rate,tag=None):
                ''' @TODO 'channels_last' is default? is this acutally a separable conv.
                W-Net : https://arxiv.org/pdf/1711.08506.pdf
                U-Net : https://arxiv.org/pdf/1505.04597.pdf
                U-ENC for W-Net should be:
                    depthwise separable conv -> depthwise separable conv -> dconv

                    One important modification in our architecture is that all of the modules use the depthwise
                    separable convolution layers introduced in U-Net except modules 1,  9,  10,  and 18.
                    A depthwise separable convolution operation consists of a depthwise  convolution  and  a
                    pointwise convolution. The idea behind such an operation is to examine spatial
                    cor-relations and cross-channel correlations independently a depthwise convolution performs
                    spatial convolutions independently over each channel and then a pointwise convolution projects
                    the feature channels by the depthwise convolution onto a new channel space.  As a consequence,
                    the network gains performance more efficiently with the same number of parameters.

                '''

                # net_3 = tf.layers.conv3d(inputs, filters, kernel, strides=[1, 1,1], dilation_rate=[1, 1, 1],
                #                          padding='SAME', activation=activation,
                #                          kernel_initializer=keras.initializers.he_normal() ,
                #                          data_format='channels_last')
                net_3 = tf.layers.conv3d_transpose(inputs,
                                                   filters,
                                                   kernel,
                                                   strides=[2, 2, 2],
                                                   padding='SAME',
                                                   activation=activation,
                                                   kernel_initializer=keras.initializers.he_normal(),
                                                   data_format='channels_last',
                                                   name=name_tag('dconv3d', tag))
                
                # net_3 =  tf.layers.max_pooling3d(net_3, pool_size=kernel,
                #                                  strides = [1, 1, 1], padding = 'SAME', data_format='channels_last')
                net_3 = tf.layers.conv3d(net_3,
                                         filters,
                                         kernel,
                                         strides=[1, 1, 1],
                                         padding='SAME',
                                         activation=activation,
                                         kernel_initializer=keras.initializers.he_normal(),
                                         data_format='channels_last', # @TODO: conv or separable conv?
                                         name=name_tag('conv3d', tag))
                
                net_3 = tf.layers.max_pooling3d(net_3,
                                                pool_size=kernel,
                                                strides=[1, 1, 1],
                                                padding='SAME',
                                                data_format='channels_last',
                                                name=name_tag('max_pooling3d', tag))
                # net_3 = tf.layers.batch_normalization(net_3, center=True, scale=True, training=phase)
                # net_3 = tf.layers.dropout(net_3,rate=0.20, training=phase)
                return net_3

            def middle_block(inputs, filters, prev_filters, kernel, activation, phase, tag=None):
                net_1 = tf.layers.conv3d(inputs,
                                         filters,
                                         kernel,
                                         strides=[1, 1, 1],
                                         dilation_rate=[1, 1, 1],
                                         padding='SAME',
                                         activation=activation,
                                         kernel_initializer=keras.initializers.lecun_normal(),
                                         data_format='channels_last',
                                         name=name_tag('conv3d_1', tag))

                # net_1 =  tf.layers.max_pooling3d(net_1, pool_size=kernel,
                #                                  strides = [1, 1, 1], padding = 'SAME', data_format='channels_last',
                #                                  name=name_tag('max_pooling3d', tag))
                net_1 = tf.layers.conv3d(net_1,
                                         filters,
                                         kernel,
                                         strides=[1, 1, 1],
                                         dilation_rate=[1, 1, 1],
                                         padding='SAME',
                                         activation=activation,
                                         kernel_initializer=keras.initializers.he_normal(),
                                         data_format='channels_last',
                                         name=name_tag('conv3d_2', tag))

                net_1 = tf.layers.max_pooling3d(net_1,
                                                pool_size=kernel,
                                                strides=[1, 1, 1],
                                                padding='SAME',
                                                name=name_tag('max_pooling3d', tag))
                # net_1 = tf.layers.batch_normalization(net_1, center=True, scale=True, training=phase)
                # net_1 = tf.layers.dropout(net_1,rate=0.20, training=phase)
                return net_1

            def middle_block_fc(inputs, filters, prev_filters, kernel, activation, phase, tag=None):
                net_1 = tf.layers.conv3d(inputs,
                                         filters,
                                         kernel,
                                         strides=[1, 1, 1],
                                         padding='SAME',
                                         activation=activation,
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name=name_tag('conv3d_1', tag))

                net_1 = tf.layers.flatten(net_1)
                # @TODO refactor hardcoded dimensions? 6, 12, 6 = 432
                net_1 = tf.layers.dense(net_1, 432,
                                        activation=activation,
                                        kernel_initializer=keras.initializers.lecun_normal(),
                                        name=name_tag('fc', tag))

                # @TODO refactor hardcoded dimensions? 6, 12, 6 = 432
                net_1 = tf.reshape(net_1,
                                   shape=(-1, 6, 12, 6, 1),
                                   name=name_tag('reshape', tag))

                net_1 = tf.layers.conv3d(net_1,
                                         prev_filters,
                                         kernel,
                                         strides=[1, 1, 1],
                                         padding='SAME',
                                         activation=activation,
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name=name_tag('conv3d_2', tag))
                # net_1 = tf.layers.batch_normalization(net_1, center=True, scale=True, training=phase
                #                                       name=name_tag('bn', tag))
                # net_1 = tf.layers.dropout(net_1,rate=0.20, training=phase
                #                                       name=name_tag('dropout', tag))
                return net_1

            def wnet(inputs, z, y, x, phase, keep_prob, params, n_input_variables):
                # encoder
                with tf.name_scope("U-encoder") as scope:
                    print('inputs {}', shape_so(inputs))
                    net_e1_1 = conv_block(inputs,
                                          filters=params.graph_params.LAYER_1,
                                          prev_filters=params.graph_params.LAYER_1,
                                          kernel=params.graph_params.KERNEL1,
                                          activation=tf.nn.leaky_relu,
                                          phase=phase,
                                          dil_rate=[1, 1, 1],
                                          tag='uenc_conv_block_1')

                    print('e1_1 {}', shape_so(net_e1_1))
                    net_e2_1 = conv_block(net_e1_1,
                                          filters=params.graph_params.LAYER_2,
                                          prev_filters=params.graph_params.LAYER_2,
                                          kernel=params.graph_params.KERNEL1,
                                          activation=tf.nn.leaky_relu,
                                          phase=phase,
                                          dil_rate=[1, 1, 1],
                                          tag='uenc_conv_block_2')

                    # print('e2_1 {}',shape_so(net_e2_1))
                    net_e3_1 = conv_block(net_e2_1,
                                          filters=params.graph_params.LAYER_3,
                                          prev_filters=params.graph_params.LAYER_3,
                                          kernel=params.graph_params.KERNEL2,
                                          activation=tf.nn.leaky_relu,
                                          phase=phase,
                                          dil_rate=[1, 1, 1],
                                          tag='uenc_conv_block_3')

                    print('e3_1 {}', shape_so(net_e3_1))

                    ## middle layer
                    # inputs,filters,prev_filters,kernel,activation,phase
                    net_m1_1 = middle_block(net_e3_1,
                                            filters=params.graph_params.LAYER_4,
                                            prev_filters=params.graph_params.LAYER_3,
                                            kernel=params.graph_params.KERNEL2,
                                            activation=tf.nn.leaky_relu,
                                            phase=phase,
                                            tag='uenc_conv_middle_block_4')
                    print('m1_1 {}',shape_so(net_m1_1))

                    # net_c3_1 = tf.concat([net_m1_1, net_e3_1], axis=-1)
                    # inputs,filters,prev_filters,kernel,activation,phase,dil_rate
                    net_d3_1 = deconv_block(net_m1_1,
                                            filters=params.graph_params.LAYER_3,
                                            prev_filters=params.graph_params.LAYER_2,
                                            kernel=params.graph_params.KERNEL2,
                                            activation=tf.nn.leaky_relu,
                                            phase=phase,
                                            dil_rate=[1, 1, 1],
                                            tag='uenc_dconv_block_5')

                    # net_c2_1 = tf.concat([net_d3_1, net_e2_1], axis=-1) #
                    net_d2_1 = deconv_block(net_d3_1,
                                            filters=params.graph_params.LAYER_2,
                                            prev_filters=params.graph_params.LAYER_1,
                                            kernel=params.graph_params.KERNEL1,
                                            activation=tf.nn.leaky_relu,
                                            phase=phase,
                                            dil_rate=[1, 1, 1],
                                            tag='uenc_dconv_block_6')

                    # net_c1_1 = tf.concat([net_d2_1,net_e1_1], axis=-1) #
                    net_d1_1 = deconv_block(net_d2_1,
                                            params.graph_params.LAYER_1,
                                            params.graph_params.LAYER_1,
                                            params.graph_params.KERNEL1,
                                            tf.nn.leaky_relu,
                                            phase,
                                            [1, 1, 1],
                                            tag='uenc_dconv_block_7')
                    # net_d1_1 = tf.layers.batch_normalization(net_d1_1, center=True, scale=True,
                    #                                          training=phase, momentum=0.90)

                    # final layer for first U
                    # net_c0_1 = tf.concat([net_d1_1,net_a], axis=-1)
                    net_feed = tf.layers.conv3d(net_d1_1,
                                                params.graph_params.N_CLASSES,
                                                params.graph_params.KERNEL2,
                                                dilation_rate=[1, 1, 1],
                                                strides=[1, 1, 1],
                                                padding='SAME',
                                                activation=tf.nn.softmax,
                                                kernel_initializer=keras.initializers.he_normal(),
                                                data_format='channels_last',
                                                name=name_tag('conv3d_final_layer','uenc_conv_block_7'))

                # net_feed = tf.nn.softmax(net_feed, axis=4)

                # decoder
                with tf.name_scope("U-decoder"):
                    net_d_1 = tf.layers.conv3d(net_feed,params.graph_params.LAYER_1,params.graph_params.KERNEL1,
                                               dilation_rate=[1, 1, 1],
                                               strides=[1, 1, 1],
                                               padding='SAME',
                                               activation=tf.nn.leaky_relu,
                                               name=name_tag('conv3d_first_layer','udec_conv_block_1'))
                    # inputs,filters,prev_filters,kernel,activation,phase,dil_rate,tag = None
                    net_de1_1 = conv_block(net_d_1,
                                           filters=params.graph_params.LAYER_1,
                                           prev_filters=params.graph_params.LAYER_1,
                                           kernel=params.graph_params.KERNEL1,
                                           activation=tf.nn.leaky_relu,
                                           phase=phase,
                                           dil_rate=[1, 1, 1],
                                           tag='udec_conv_block_1')

                    # net_bridge1 = tf.concat([net_de1_1,net_e1_1], axis=-1)
                    net_de2_1 = conv_block(net_de1_1,
                                           filters=params.graph_params.LAYER_2,
                                           prev_filters=params.graph_params.LAYER_2,
                                           kernel=params.graph_params.KERNEL1,
                                           activation=tf.nn.leaky_relu,
                                           phase=phase,
                                           dil_rate=[1, 1, 1],
                                           tag='udec_conv_block_2')

                    # net_bridge2 = tf.concat([net_de2_1,net_e2_1], axis=-1)
                    net_de3_1 = conv_block(net_de2_1,
                                           params.graph_params.LAYER_3,
                                           params.graph_params.LAYER_3,
                                           params.graph_params.KERNEL2,
                                           tf.nn.leaky_relu,
                                           phase,
                                           [1, 1, 1],
                                           tag='udec_conv_block_3')
                    # net_bridge3 = tf.concat([net_de3_1,net_e3_1], axis=-1)

                    ## middle layer
                    # net_cA = tf.concat([net_de3_1, net_m1_1], axis=-1)
                    net_dm1_1 = middle_block(net_de3_1,
                                             params.graph_params.LAYER_4,
                                             params.graph_params.LAYER_3,
                                             params.graph_params.KERNEL2,
                                             tf.nn.leaky_relu,
                                             phase,
                                             tag='udec_conv_middle_block_4')

                    # net_dc3_1 = tf.concat([net_dm1_1, net_de3_1], axis=-1) # #net_e3_1 , net_m1_1
                    net_dd3_1 = deconv_block(net_dm1_1,
                                             params.graph_params.LAYER_3,
                                             params.graph_params.LAYER_2,
                                             params.graph_params.KERNEL2,
                                             tf.nn.leaky_relu,
                                             phase,
                                             [1, 1, 1],
                                             tag='udec_dconv_block_5')

                    # net_dc2_1 = tf.concat([net_dd3_1, net_de2_1], axis=-1) # #net_e2_1, , neprint(total_out[1].get_shape())t_d3_1
                    net_dd2_1 = deconv_block(net_dd3_1,
                                             params.graph_params.LAYER_2,
                                             params.graph_params.LAYER_1,
                                             params.graph_params.KERNEL1,
                                             tf.nn.leaky_relu,
                                             phase,
                                             [1, 1, 1],
                                             tag='udec_dconv_block_6')

                    # net_dc1_1 = tf.concat([net_dd2_1,net_de1_1], axis=-1) # #net_e1_1 , net_d2_1
                    net_dd1_1 = deconv_block(net_dd2_1,
                                             params.graph_params.LAYER_1,
                                             params.graph_params.LAYER_1,
                                             params.graph_params.KERNEL1,
                                             tf.nn.leaky_relu,
                                             phase,
                                             [1, 1, 1],
                                             tag='udec_dconv_block_7')
                    # net_dd1_1 = tf.layers.batch_normalization(net_dd1_1, center=True, scale=True, training=phase, momentum=0.90)

                    # final layer for second U
                    # net_t = tf.concat([net_dd1_1, net_feed], axis=-1)

                    net_r = tf.layers.conv3d(net_dd1_1,
                                             n_input_variables,
                                             params.graph_params.KERNEL2,
                                             dilation_rate=[1, 1, 1],
                                             strides=[1, 1, 1],
                                             padding='SAME',
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer=keras.initializers.he_normal(),
                                             name=name_tag('conv3d_final_layer','udec_conv_block_7'))

                return net_feed, net_r

            # Network
            total_out = wnet(self.input_processed,
                             params.image_params.CROP_PAD_IMAGE_Z,
                             params.image_params.CROP_PAD_IMAGE_Y,
                             params.image_params.CROP_PAD_IMAGE_X,
                             self.phase,
                             self.keep_prob,
                             params,
                             self.n_input_variables)

            print(total_out[0].get_shape())
            print(total_out[1].get_shape())

            wnet_categories = total_out[0]
            predictionsOut_1 = tf.argmax(wnet_categories, axis=4)
            predictionsOut_1 = tf.expand_dims(predictionsOut_1, axis=4)

            predictionsOut_1 = tf.reshape(predictionsOut_1,
                                          shape=(-1,
                                                 params.image_params.CROP_PAD_IMAGE_X,
                                                 params.image_params.CROP_PAD_IMAGE_Y,
                                                 params.image_params.CROP_PAD_IMAGE_Z,
                                                 1))
            # @TODO is this the how depthwise separable conv is being handled?
            predictionsOut_1 = tf.transpose(predictionsOut_1, perm=[0, 3, 2, 1, 4]) # @TODO Parameterize these
            predictionsOut_1 = tf_crop_or_pad_along_axis(predictionsOut_1, self.org_z, 1)
            predictionsOut_1 = tf_crop_or_pad_along_axis(predictionsOut_1, self.org_y, 2)
            predictionsOut_1 = tf_crop_or_pad_along_axis(predictionsOut_1, self.org_x, 3)
            # predictionsOut_1 = tf.transpose(predictionsOut_1, perm = [0,3,2,1,4])
            predictionsOut_1 = tf.reshape(predictionsOut_1, shape=(-1,
                                                                   self.org_z,
                                                                   self.org_y,
                                                                   self.org_x))

            predictionsOut = tf.reshape(tf.cast(predictionsOut_1, tf.int64),
                                        shape=(-1, self.org_x * self.org_y * self.org_z),
                                        name='outputs')

            probs = tf.identity(wnet_categories,name='probs')  # tf.nn.softmax(wnet_categories, axis=2, name = 'probs')

            ## decoder - output
            wnet_original = tf.identity(total_out[1],name='decoder_outputs')

            original_image = tf.identity(self.input_processed,name='encoder_inputs')
            unprocessed_image = tf.identity(self.input_images,name='process_inputs')
            loss_dec = tf.losses.mean_squared_error(self.input_processed,wnet_original)
            print(loss_dec.get_shape())

            loss_enc = tf.map_fn(elems=np.arange(params.runtime_params.BATCH_SIZE),
                                 fn=lambda i: centroids_similarity_loss(self.input_processed[i, :, :, :, :],
                                                                        wnet_categories[i, :, :, :, :],
                                                                        params.image_params.CROP_PAD_IMAGE_Z,
                                                                        params.image_params.CROP_PAD_IMAGE_Y,
                                                                        params.image_params.CROP_PAD_IMAGE_X,
                                                                        params.graph_params.N_CLASSES,
                                                                        self.n_input_variables),
                                 dtype=(tf.float32))
            print(loss_enc.get_shape())

            self.loss_decf = tf.reduce_sum(loss_dec)  # + tf.reduce_sum(loss_enc)
            self.loss_encf = tf.reduce_sum(loss_enc)
            # Training operations
            learning_rate = tf.train.cosine_decay_restarts(learning_rate=0.0001,
                                                           global_step=global_step,
                                                           first_decay_steps=10,
                                                           t_mul=2.0,
                                                           m_mul=1.0,
                                                           alpha=0.01)
            trainer = tf.train.AdamOptimizer(learning_rate)
            self.saver = tf.train.Saver(max_to_keep=1000)
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                # self.training_step_add = trainer.minimize(self.loss_misc)
                self.training_step_enc = trainer.minimize(self.loss_encf)
                self.training_step_dec = trainer.minimize(self.loss_decf)

            ## end graph

    def trainGraph(self,params,config,datatopulltrain,datatopulltest,variables):
        """
        Trains the segmentation
        """
        varsM = variables
        with tf.Session(graph=self.graph,config=config) as session:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            # session.graph.finalize()
            # tf.train.start_queue_runners(sess=session, start = True)
            result = []
            best_test_error_Image = 100000000.0
            best_test_error_Cluster = 100000000.0
            best_mult = 100.0
            counter2 = 1.0
            for e in range(1,params.runtime_params.N_EPOCHS + 1):
                counter = 0
                vtk.vtkObject.GlobalWarningDisplayOff()
                total_train_loss = 0.0
                total_test_loss = 0.0
                total_batch = params.runtime_params.NUM_ITERATIONS
                for i in range(1,params.runtime_params.NUM_ITERATIONS):
                    train_x = get_batches_fn_random(params.runtime_params.BATCH_SIZE,
                                                    datatopulltrain,
                                                    params,
                                                    varsM,
                                                    i,
                                                    'Train')

                    op2,train_loss = session.run([self.training_step_dec,self.loss_decf],
                                                 feed_dict={self.input_images: train_x,
                                                            self.phase: True,
                                                            self.keep_prob: 1.0,
                                                            self.x_voxels: 52,
                                                            self.y_voxels: 336,
                                                            self.z_voxels: 65})


                    test_x = get_batches_fn_random(params.runtime_params.BATCH_SIZE,
                                                   datatopulltest,
                                                   params,
                                                   varsM,
                                                   i,
                                                   'Test')

                    test_loss = session.run(self.loss_encf, feed_dict={self.input_images: test_x,
                                                                       self.phase: False,
                                                                       self.keep_prob: 1.0,
                                                                       self.x_voxels: 52,
                                                                       self.y_voxels: 336,
                                                                       self.z_voxels: 65})
                    total_train_loss += train_loss
                    if (i % 10 == 0):
                        counter+=1

                        total_test_loss += test_loss
                        op3 = session.run([self.training_step_enc],
                                          feed_dict={self.input_images: train_x,
                                                     self.phase: True,
                                                     self.keep_prob: 1.0,
                                                     self.x_voxels: 52,
                                                     self.y_voxels: 336,
                                                     self.z_voxels: 65})

                        print('batch {} - Train loss: {} -Test loss: {}'.format(i, np.round(total_train_loss/np.float(i),3), np.round(total_test_loss/np.float(counter),3)))
                        if (i % 30 == 0):
                            self.saver.save(session, params.runtime_params.MODEL_PATH + '_epoch_' + str(e) + '_batch_' + str(i))
                            tf.train.write_graph(session.graph_def, '.',params.runtime_params.MODEL_PATH + '_epoch_' + str(e) + '_batch_' + str(i) + '.pb', False)
                            result.append([e, i, np.round(total_train_loss / np.float(i), 3), np.round(total_test_loss / np.float(counter), 3)])
                            np.savetxt(params.runtime_params.MODEL_PATH + '_test.csv', result, delimiter=',')
                        mult = np.round(total_test_loss / np.float(counter), 3) * np.round(total_train_loss / np.float(i), 3)

                        if (mult < best_mult):  # (np.round(total_test_loss/np.float(counter),3)<best_test_error_Cluster and total_train_loss/np.float(i) <= best_test_error_Image ):
                            best_test_error_Cluster = np.round(total_test_loss / np.float(counter), 3)
                            best_test_error_Image = total_train_loss / np.float(i)
                            best_mult = best_test_error_Cluster * best_test_error_Image
                            print (best_test_error_Cluster)
                            self.saver.save(session, params.runtime_params.MODEL_PATH + 'bestModel')
                            tf.train.write_graph(session.graph_def, '.', params.runtime_params.MODEL_PATH + 'bestModel' + '.pb', False)
                        # del train_x, test_x, train_loss, #op2#, op3
                    gc.collect()
                    counter2+=1
                    print('Epoch {} -Train loss: {} -Test loss: {}'.format(e,np.round(total_train_loss / np.float(i),3), np.round(test_loss / np.float(counter2), 3)))
                    result.append([e, i, np.round(total_train_loss / np.float(i), 3), np.round(total_test_loss / np.float(counter2), 3)])
                    np.savetxt(params.runtime_params.MODEL_PATH + 'test.csv', result, delimiter=',')
                    gc.collect()

        ############### main function #####################


def main(yaml_path):
    params = read_params(yaml_path)

    datatopull = pd.read_csv('MergedListCleanNFS.csv')
    datatopull = datatopull.apply(np.random.permutation,axis=0)
    print(len(np.unique(datatopull.run)))
    train = np.unique(np.random.choice(datatopull.run,size=np.int(len(datatopull.run) * 0.93),replace=False))
    print(len(train))
    test = np.array(np.setdiff1d(datatopull.run,train))
    n_in_train = len(train)
    print(len(train))
    train = np.random.choice(train,
                             size=(n_in_train // params.runtime_params.BATCH_SIZE) * params.runtime_params.BATCH_SIZE,
                             replace=False)
    print(len(train))
    print(len(datatopull.run))
    datatopulltrain = datatopull[datatopull.run.isin(train)]

    print(len(datatopulltrain.run))
    datatopulltest = datatopull[datatopull.run.isin(test)]

    total_batch_train = len(datatopulltrain['run'].values)
    total_batch_test = len(datatopulltest['run'].values)
    print(total_batch_train)
    print(total_batch_test)

    params.runtime_params.NUM_ITERATIONS = total_batch_train // params.runtime_params.BATCH_SIZE
    print(params.runtime_params.NUM_ITERATIONS)

    variables = params.runtime_params.VARIABLES

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,allow_growth=True)
    config = tf.ConfigProto(intra_op_parallelism_threads=8,  # multiprocessing.cpu_count(),
                            inter_op_parallelism_threads=8,  # multiprocessing.cpu_count(),
                            log_device_placement=True,
                            gpu_options=gpu_options,
                            allow_soft_placement=True,
                            device_count={'GPU': 1,
                                          'CPU': 8}
                            )
    graph = WNetTensorflow(params,variables)
    graph.trainGraph(params,config,datatopulltrain,datatopulltest,variables)
    return graph


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(**vars(args))

