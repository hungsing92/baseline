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

def mobilenet(inputs,
          num_classes=1000,
          is_training=True,
          width_multiplier=1,
          scope='MobileNet'):
  """ MobileNet
  More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """

  def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                width_multiplier,
                                sc,
                                downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn

  with tf.variable_scope(scope) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=None,
                        outputs_collections=[end_points_collection]):
      with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          activation_fn=tf.nn.relu):
        net = slim.convolution2d(inputs, round(32 * width_multiplier), [3,3], stride=1, padding='SAME', scope='conv_1')
        net = slim.batch_norm(net, scope='conv_1/batch_norm')
        net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
        net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
        net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
        net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
        net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
        net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')

        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')

        net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=False, sc='conv_ds_13')
        net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
        # net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')

    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    # net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    # end_points['squeeze'] = net
    # logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc_16')
    # predictions = slim.softmax(logits, scope='Predictions')

    # end_points['Logits'] = logits
    # end_points['Predictions'] = predictions

  return end_points

mobilenet.default_image_size = 224


def mobilenet_arg_scope(weight_decay=0.0):
  """Defines the default mobilenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the MobileNet model.
  """
  with slim.arg_scope(
      [slim.convolution2d, slim.separable_convolution2d],
      weights_initializer=slim.initializers.xavier_initializer(),
      biases_initializer=slim.init_ops.zeros_initializer(),
      weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
    return sc


def top_feature_net(input, anchors, inds_inside, num_bases):
  stride=8
  with tf.variable_scope("top_base"):
    arg_scope=mobilenet_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
      end_points=mobilenet(input,num_classes=1000,is_training=True,width_multiplier=1,scope='MobileNet')
      block=end_points['top_base/MobileNet/conv_ds_14/depthwise_conv']
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
                                       nms_thresh=0.7, min_size=stride, nms_pre_topn=500, nms_post_topn=50,
                                       name ='nms')
  
    #<todo> feature = upsample2d(block, factor = 4,  ...)
    feature = block
  
    # print ('top: scale=%f, stride=%d'%(1./stride, stride))
    return feature, scores, probs, deltas, rois, roi_scores
    
def rgb_feature_net(input):
  # with tf.variable_scope("rgb_base"):
    arg_scope=mobilenet_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
      end_points=mobilenet(input,num_classes=1000,is_training=True,width_multiplier=1,scope='MobileNet')
      block=end_points['MobileNet/conv_ds_14/depthwise_conv']
    #<todo> feature = upsample2d(block, factor = 4,  ...)
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

        roi_features,  roi_idxs = tf_roipooling(feature,roi, pool_height, pool_width, pool_scale, name='%d/pool'%n)
        roi_features = flatten(roi_features)
        if input is None:
            input = roi_features
        else:
            input = concat([input,roi_features], axis=1, name='%d/cat'%n)

  with tf.variable_scope('fuse-block-1') as scope:
    block = linear_bn_relu(input, num_hiddens=512, name='1')
    block = linear_bn_relu(block, num_hiddens=512, name='2')
    block = linear_bn_relu(block, num_hiddens=512, name='3')
    block = linear_bn_relu(block, num_hiddens=512, name='4')

  #include background class
  with tf.variable_scope('fuse') as scope:
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

