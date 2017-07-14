from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from net.utility.file import *
from net.blocks import *
from net.rpn_nms_op import tf_rpn_nms
from net.roipooling_op import roi_pool as tf_roipooling
import pdb
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import vgg

keep_prob=1
nms_pre_topn_=500
nms_post_topn_=200
is_training=False

def top_feature_net(input, anchors, inds_inside, num_bases):
  stride=4
  with tf.variable_scope("top_base") as sc:
    arg_scope = resnet_v1.resnet_arg_scope(is_training=is_training)
    with slim.arg_scope(arg_scope):
      net, end_points = resnet_v1.resnet_v1_50(input, None, global_pool=False, output_stride=8)
      block4=end_points['top_base/resnet_v1_50/block4']
      block3=end_points['top_base/resnet_v1_50/block3']
      block2=end_points['top_base/resnet_v1_50/block2']
      tf.summary.histogram('top_block4', block4)
      tf.summary.histogram('top_block3', block3)
      tf.summary.histogram('top_block2', block2)
  with tf.variable_scope("top_up") as sc:
    block4_   = conv2d_relu(block4, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='4')
    up4     = upsample2d(block4_, factor = 2, has_bias=True, trainable=True, name='up4')
    block3_   = conv2d_relu(block3, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='3')
    up3     = upsample2d(block3_, factor = 2, has_bias=True, trainable=True, name='up3')
    block2_   = conv2d_relu(block2, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='2')
    up2     = upsample2d(block2_, factor = 2, has_bias=True, trainable=True, name='up2')
    up_34      =tf.add(up4, up3, name="up_add_3_4")
    up      =tf.add(up_34, up2, name="up_add_3_4_2")
    block    = conv2d_relu(up, num_kernels=256, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='rgb_ft')
  with tf.variable_scope('top') as scope:
    up      = conv2d_relu(block, num_kernels=256, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
    scores  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='score')
    probs   = tf.nn.softmax( tf.reshape(scores,[-1,2]), name='prob')
    deltas  = conv2d(up, num_kernels=4*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='delta')

  #<todo> flip to train and test mode nms (e.g. different nms_pre_topn values): use tf.cond
  with tf.variable_scope('top-nms') as scope:    #non-max
    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()
    img_scale = 1
    # pdb.set_trace()
    rois, roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside,
                                     stride, img_width, img_height, img_scale,
                                     nms_thresh=0.7, min_size=stride, nms_pre_topn=nms_pre_topn_, nms_post_topn=nms_post_topn_,
                                     name ='nms')

  #<todo> feature = upsample2d(block, factor = 4,  ...)
  feature = block
    
      # print ('top: scale=%f, stride=%d'%(1./stride, stride))
  return feature, scores, probs, deltas, rois, roi_scores
    
def rgb_feature_net(input):

    arg_scope = resnet_v1.resnet_arg_scope(is_training=is_training)
    with slim.arg_scope(arg_scope):
      net, end_points = resnet_v1.resnet_v1_50(input, None, global_pool=False, output_stride=8)
      # pdb.set_trace()
      block4=end_points['resnet_v1_50/block4']
      block3=end_points['resnet_v1_50/block3']
      block2=end_points['resnet_v1_50/block2']
      tf.summary.histogram('rgb_block4', block4)
      tf.summary.histogram('rgb_block3', block3)
      tf.summary.histogram('rgb_block2', block2)
      with tf.variable_scope("rgb_up") as sc:
        block4_   = conv2d_relu(block4, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='4')
        up4     = upsample2d(block4_, factor = 2, has_bias=True, trainable=True, name='up4')
        block3_   = conv2d_relu(block3, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='3')
        up3     = upsample2d(block3_, factor = 2, has_bias=True, trainable=True, name='up3')
        block2_   = conv2d_relu(block2, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='2')
        up2     = upsample2d(block2_, factor = 2, has_bias=True, trainable=True, name='up2')
        up_34      =tf.add(up4, up3, name="up_add_3_4")
        up      =tf.add(up_34, up2, name="up_add_3_4_2")
        # up      = tf.multiply(up, 1/3000000)
        block    = conv2d_relu(up, num_kernels=256, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='rgb_ft')
        # block1_   = conv2d_bn_relu(block1, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='1')
        # up      =tf.add(block1_, up_, name="up_add")
      # block   = conv2d_bn_relu(block, num_kernels=512, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='2')
    #<todo> feature = upsample2d(block, factor = 4,  ...)
      tf.summary.histogram('rgb_top_block', block)

    feature = block
    return feature

#------------------------------------------------------------------------------
def front_feature_net(input):
  feature = None
  return feature


#------------------------------------------------------------------------------
def fusion_net(feature_list, num_class, out_shape=(8,3)):
  stride=4
  num=len(feature_list)

  input = None
  with tf.variable_scope('fuse-input') as scope:
    for n in range(num):
        feature     = feature_list[n][0]
        roi         = feature_list[n][1]
        pool_height = feature_list[n][2]
        pool_width  = feature_list[n][3]
        pool_scale  = feature_list[n][4]
        if (pool_height==0 or pool_width==0): continue
        # feature   = conv2d_bn_relu(feature, num_kernels=512, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='%d'%n)
        roi_features,  roi_idxs = tf_roipooling(feature,roi, pool_height, pool_width, pool_scale, name='%d/pool'%n)
        # pdb.set_trace()
        roi_features=flatten(roi_features)
        with tf.variable_scope('fuse-block-1-%d'%n):
          tf.summary.histogram('fuse-block_input_%d'%n, roi_features)
          block = linear_bn_relu(roi_features, num_hiddens=1024, name='1')#512, so small?
          tf.summary.histogram('fuse-block1_%d'%n, block)
          block = tf.nn.dropout(block, keep_prob, name='drop1')
  
        if input is None:
            input = block
        else:
            input = concat([input,block], axis=1, name='%d/cat'%n)

        # if input is None:
        #     input = roi_features
        # else:
        #     input = concat([input,roi_features], axis=1, name='%d/cat'%n)

  # with tf.variable_scope('fuse-block-1') as scope:
  #   input_=None
  #   for i in len(feat):
  #     with tf.variable_scope('fc%d'%i):
  #       tf.summary.histogram('fuse-block_input_%d'%i, feat[i])
  #       block = linear_bn_relu(feat[i], num_hiddens=4096, name='1')#512, so small?
  #       tf.summary.histogram('fuse-block1_%d'%i, block)
  #       block = tf.nn.dropout(block, keep_prob, name='drop1')
  #       block = linear_bn_relu(block, num_hiddens=4096, name='2')
  #       tf.summary.histogram('fuse-block2_%d'%i, block)
  #       block = tf.nn.dropout(block, keep_prob, name='drop2')
  #       if input_ is None:
  #           input_ = block
  #       else:
  #           input_ = concat([input_,block], axis=1, name='%d/cat'%n)
    # block = linear_bn_relu(block, num_hiddens=512, name='3')
    # block = linear_bn_relu(block, num_hiddens=512, name='4')

  #include background class
  with tf.variable_scope('fuse') as scope:
    block = linear_bn_relu(input, num_hiddens=512, name='4')#512, so small?
    # block = tf.nn.dropout(block, keep_prob, name='drop4')
    with tf.variable_scope('3D') as sc:
      dim = np.product([*out_shape])
      scores_3d  = linear(block, num_hiddens=num_class,     name='score')
      probs_3d   = tf.nn.softmax (scores_3d, name='prob')
      deltas_3d  = linear(block, num_hiddens=dim*num_class, name='box')
      deltas_3d  = tf.reshape(deltas_3d,(-1,num_class,*out_shape))
    with tf.variable_scope('2D') as sc_:
      dim = np.product(4)
      deltas_2d  = linear(block, num_hiddens=dim*num_class, name='box')
      deltas_2d  = tf.reshape(deltas_2d,(-1,num_class,4))


  return  scores_3d, probs_3d, deltas_3d, deltas_2d


# main ###########################################################################
# to start in tensorboard:
#    /opt/anaconda3/bin
#    ./python tensorboard --logdir /root/share/out/didi/tf
#     http://http://localhost:6006/    

