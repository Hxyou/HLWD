import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import h5py
import socket
import importlib
import os
import os.path as osp
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'config'))
sys.path.append(os.path.join(BASE_DIR, 'dataset'))
from dataset import provider
from utils import config
from dataset import data_util
cfg = config.config()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

BATCH_SIZE = cfg.batch_size
NUM_POINT = cfg.ps_input_num
MAX_EPOCH = cfg.max_epoch
BASE_LEARNING_RATE = cfg.lr
GPU_INDEX = cfg.gpu
MOMENTUM = cfg.momentum
OPTIMIZER = cfg.optimizer
DECAY_STEP = cfg.decay_step
DECAY_RATE = cfg.decay_rate
NUM_CLASSES = cfg.class_num
NUM_VIEWS = cfg.views_num
IMG_SIZE = cfg.img_size
LOAD_DIR = cfg.ckpt_view_model

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

MODEL = importlib.import_module(cfg.model)
MODEL_FILE = os.path.join(BASE_DIR, 'models', cfg.model+'.py')
LOG_DIR = cfg.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp config/config.cfg %s' % (LOG_DIR)) # bkp of cfg def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
SAVE_DIR = cfg.ckpt_folder


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


def get_learning_rate_all(batch):
    learning_rate = tf.train.exponential_decay(
                        0.0001,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        15000,          # Decay step.
                        0.1,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


# def get_learning_rate_all(batch):
#     learning_rate = tf.train.exponential_decay(
#                         BASE_LEARNING_RATE,  # Base learning rate.
#                         batch * BATCH_SIZE,  # Current index into the dataset.
#                         DECAY_STEP,          # Decay step.
#                         DECAY_RATE,          # Decay rate.
#                         staircase=True)
#     learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
#     return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():

    dataset_test = provider.pc_view_data(cfg, state='test',
                                         batch_size=cfg.batch_size, shuffle=True,
                                         img_sz=cfg.img_size, ps_input_num=cfg.ps_input_num)

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, images_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_VIEWS, IMG_SIZE,  NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, trainable=False)
            batch_all = tf.Variable(0, trainable=False)

            # batch_mv = tf.Variable(0, trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            # pred_pc, pred_mv, fc6_b, fc6, view_pooling = MODEL.multi_modal(pointclouds_pl, images_pl,  is_training_pl, bn_decay=bn_decay)
            # loss_pc = MODEL.get_loss_pc(pred_pc, labels_pl)
            # loss_mv = MODEL.get_loss_mv(pred_mv, labels_pl)
            # loss = 1 * loss_mv + loss_pc
            pred_pc = MODEL.multi_modal(pointclouds_pl, images_pl,
                                   is_training_pl, bn_decay=bn_decay)


            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)


        # Init variables and load pre-trained mvcnn
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: False})
        saver.restore(sess, save_path=osp.join(cfg.ckpt_folder, '10_model.ckpt'))

        ops = {'pointclouds_pl': pointclouds_pl,
               'images_pl': images_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred_pc':pred_pc,
               'step': batch}

        test(sess, ops, dataset_test)


def test(sess, ops, dataset_test):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct_pc = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    num_data, num_batches = dataset_test.get_len()

    print ('num of test data: %d' % num_data)
    print ('num of test batch: %d' % num_batches)

    for batch_idx in range(num_batches):
        imgs, pcs, lbls = dataset_test.get_batch(idx=batch_idx, point_num=NUM_POINT, view_num=NUM_VIEWS)

        feed_dict = {ops['pointclouds_pl']: pcs,
                     ops['images_pl']: imgs,
                     ops['labels_pl']: lbls,
                     ops['is_training_pl']: is_training}
        pred_pc = sess.run(ops['pred_pc'], feed_dict=feed_dict)
        # print(ft.shape)
        pred_pc = np.argmax(pred_pc, 1)
        correct_pc = np.sum(pred_pc == lbls)

        # print('pred pc: ', pred_pc)
        # print('gt_lbl:', lbls)
        total_correct_pc += correct_pc
        total_seen += BATCH_SIZE

        for i in range(BATCH_SIZE):
            l = lbls[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_pc[i] == l)

        print('%d/%d acc %f'%(batch_idx, num_batches, total_correct_pc/float(total_seen)))

        print('pc eval avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
