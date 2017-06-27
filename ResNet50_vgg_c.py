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
from tensorflow.contrib.slim.python.slim.nets import vgg

keep_prob=0.5
nms_pre_topn_=5000
nms_post_topn_=2000
def top_feature_net(input, anchors, inds_inside, num_bases):
  stride=4
    # arg_scope = resnet_v1.resnet_arg_scope(weight_decay=0.0)
    # with slim.arg_scope(arg_scope) :
  with slim.arg_scope(vgg.vgg_arg_scope()):
    # net, end_points = resnet_v1.resnet_v1_50(input, None, global_pool=False, output_stride=8)
    block5, end_points = vgg.vgg_16(input)
    # pdb.set_trace()
    block3 = end_points['vgg_16/conv3/conv3_3']
    block4 = end_points['vgg_16/conv4/conv4_3']
  with tf.variable_scope("top_base") as sc:
    block5_   = conv2d_bn_relu(block5, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='5')
    up5     = upsample2d(block5_, factor = 2, has_bias=True, trainable=True, name='up1')
    block4_   = conv2d_bn_relu(block4, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='4')
    up4     = upsample2d(block4_, factor = 2, has_bias=True, trainable=True, name='up2')
    up_      =tf.add(up4, up5, name="up_add")
    block3_   = conv2d_bn_relu(block3, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='3')
    up       = tf.add(up_,block3_, name='fpn_up_')
    block    = conv2d_bn_relu(up, num_kernels=256, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='fpn_up')
    tf.summary.histogram('rpn_top_block', block) 
    # tf.summary.histogram('rpn_top_block_weights', tf.get_collection('2/conv_weight')[0])
    with tf.variable_scope('top') as scope:
      #up     = upsample2d(block, factor = 2, has_bias=True, trainable=True, name='1')
      #up     = block
      up      = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
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

    arg_scope = resnet_v1.resnet_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
      net, end_points = resnet_v1.resnet_v1_50(input, None, global_pool=False, output_stride=8)
      block=end_points['resnet_v1_50/block4']
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
  stride=8
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
          block = linear_bn_relu(roi_features, num_hiddens=512, name='1')#512, so small?
          tf.summary.histogram('fuse-block1_%d'%n, block)
          block = tf.nn.dropout(block, keep_prob, name='drop1')
          block = linear_bn_relu(block, num_hiddens=512, name='2')
          tf.summary.histogram('fuse-block2_%d'%n, block)
          block = tf.nn.dropout(block, keep_prob, name='drop2')
          block = linear_bn_relu(roi_features, num_hiddens=512, name='3')#512, so small?
          block = tf.nn.dropout(block, keep_prob, name='drop3')
          

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
    block = tf.nn.dropout(block, keep_prob, name='drop4')
    dim = np.product([*out_shape])
    scores  = linear(block, num_hiddens=num_class,     name='score')
    probs   = tf.nn.softmax (scores, name='prob')
    deltas  = linear(block, num_hiddens=dim*num_class, name='box')
    deltas  = tf.reshape(deltas,(-1,num_class,*out_shape))

  return  scores, probs, deltas


# main ###########################################################################
# to start in tensorboard:
#    /opt/anaconda3/bin
#    ./python tensorboard --logdir /root/share/out/didi/tf
#     http://http://localhost:6006/    

