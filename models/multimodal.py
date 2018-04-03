import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('../')
import utils.tf_util as tf_util
from transform_nets import input_transform_net
import MVCNN

def placeholder_inputs(batch_size, views, img_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  images_pl = tf.placeholder(tf.float32, shape=(batch_size, views, img_size, img_size, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, images_pl, labels_pl


def trans_net(point_cloud, k=20, is_training=False, bn_decay=None, scope='transform_net1'):
    # input (B, N, 3)  return(B, N, 3)
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

    with tf.variable_scope(scope) as sc:
        transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)

    point_cloud_transformed = tf.matmul(point_cloud, transform)

    return point_cloud_transformed


def residual_attn_block(point_cloud, mv_fc,  k=20, C_out=64, C_attn=256, is_training=True, bn_decay=None, scope='attn1_'):
    """

    :param point_cloud: (N, P, 1, C_in)
    :param mv_fc: (N, C(1024))
    :param is_training: training state
    :param k: k neighbors
    :param C_out: output channel
    :param bn_decay: bn decay
    :param scope: scope name
    :return: (N, P, 1, C_out)
    """

    point_cloud_sq = tf.squeeze(point_cloud)
    batch_size = point_cloud_sq.shape[0].value
    num_points = point_cloud_sq.shape[1].value
    num_dims_in = point_cloud_sq.shape[2].value

    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
    res1 = tf_util.conv2d(edge_feature, C_out, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope=scope+'_res1', bn_decay=bn_decay)
    res1 = tf.reduce_max(res1, axis=-2, keep_dims=True)

    # res2 = tf_util.conv2d(point_cloud, C_attn, [1,1],
    #                       padding='VALID', stride=[1, 1],
    #                       bn=True, is_training=is_training,
    #                       scope=scope+'res2_conv1', bn_decay=bn_decay)
    # res2_global = tf_util.max_pool2d(res2, [num_points, 1], padding='VALID', scope=scope+'maxpool')
    # print (res2_global.get_shape())
    res2_global = tf.expand_dims(tf.expand_dims(mv_fc, axis=1), axis=1)
    res2_global = tf.tile(res2_global, [1, num_points, 1, 1])
    # print (res2_global.get_shape())
    res2_concat = tf.concat([res2_global, point_cloud], axis=-1)
    res2_out = tf_util.conv2d(res2_concat, C_out, [1,1],
                              padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training,
                              scope=scope + '_res2', bn_decay=bn_decay)
    res2_mask = tf.nn.sigmoid(tf.log(res2_out))
    # print (res2_mask.get_shape())

    res2_attn = tf.multiply(res2_mask, res1)
    # print (res2_attn.get_shape())

    net = tf.add(res2_attn, res1)
    # print (net.get_shape())

    return net




def multi_modal(point_cloud, view_images, is_training=False, bn_decay=None, n_classes=40):
    """

    :param point_cloud: (B, N, 3)
    :param view_images: (B, V, W, H, C)
    :param is_training: is_training for dropout and bn
    :param bn_decay: bn_decay
    :param n_classes: 40
    :return: multi-modal logit and mvcnn logit
    """

    fc6_b = MVCNN.inference_multiview(view_images, n_classes, is_training=is_training)

    mv_global = tf_util.fully_connected(fc6_b, 1024, bn=True, is_training=is_training,
                                  scope='pc_mv_t', bn_decay=bn_decay)

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    k = 20

    point_cloud_transformed = trans_net(point_cloud, k=k, is_training=is_training, bn_decay=bn_decay, scope='pc_transform_net1')


    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='pc1', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net


    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='pc2', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net


    net = residual_attn_block(net, mv_global, k=k, C_out=64,
                              C_attn=256, is_training=is_training,
                              bn_decay=bn_decay, scope='pc_attn_1')
    net3 = net


    net = residual_attn_block(net, mv_global, k=k, C_out=128,
                              C_attn=256, is_training=is_training,
                              bn_decay=bn_decay, scope='pc_attn_2')
    net4 = net


    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='pc_agg', bn_decay=bn_decay)

    net = tf.reduce_max(net, axis=1, keep_dims=True)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf.concat([net, mv_global], axis=-1)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='pc_fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='pc_dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='pc_fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='pc_dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='pc_fc3')

    return net


def get_loss_pc(pred, label):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=40)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss

def get_loss_mv(pred, label):
  """ pred: B*NUM_CLASSES,
      label: B, """

  labels = tf.one_hot(indices=label, depth=40)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred)
  classify_loss = tf.reduce_mean(loss)

  # labels = tf.one_hot(indices=label, depth=40)
  # loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  # classify_loss = tf.reduce_mean(loss)
  return classify_loss


