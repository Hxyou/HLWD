import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
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

# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='dgcnn', help='Model name: dgcnn')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
# parser.add_argument('--max_epoch', type=int, default=300, help='Epoch to run [default: 250]')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
# parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
# parser.add_argument('--decay_step', type=int, default=240000, help='Decay step for lr decay [default: 200000]')
# parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
# FLAGS = parser.parse_args()


# BATCH_SIZE = FLAGS.batch_size
# NUM_POINT = FLAGS.num_point
# MAX_EPOCH = FLAGS.max_epoch
# BASE_LEARNING_RATE = FLAGS.learning_rate
# GPU_INDEX = FLAGS.gpu
# MOMENTUM = FLAGS.momentum
# OPTIMIZER = FLAGS.optimizer
# DECAY_STEP = FLAGS.decay_step
# DECAY_RATE = FLAGS.decay_rate
#
# MODEL = importlib.import_module(FLAGS.model) # import network module
# MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
# LOG_DIR = FLAGS.log_dir
# if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
# os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
# LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
# LOG_FOUT.write(str(FLAGS)+'\n')
#
# MAX_NUM_POINT = 2048
# NUM_CLASSES = 40
#
# BN_INIT_DECAY = 0.5
# BN_DECAY_DECAY_RATE = 0.5
# BN_DECAY_DECAY_STEP = float(DECAY_STEP)
# BN_DECAY_CLIP = 0.99
#
# HOSTNAME = socket.gethostname()
#
# # ModelNet40 official train/test split
# TRAIN_FILES = provider.getDataFiles( \
#     os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048/train_files.txt'))
# TEST_FILES = provider.getDataFiles(\
#     os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048/test_files.txt'))


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


# def get_learning_rate_mv(batch):
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

    dataset_train = provider.pc_view_data(cfg, state='train',
                                          batch_size=cfg.batch_size, shuffle=True,
                                          img_sz=cfg.img_size, ps_input_num=cfg.ps_input_num)

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
            # batch_mv = tf.Variable(0, trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred_pc, pred_mv = MODEL.multi_modal(pointclouds_pl, images_pl,  is_training_pl, bn_decay=bn_decay)
            loss_pc = MODEL.get_loss_pc(pred_pc, labels_pl)
            loss_mv = MODEL.get_loss_mv(pred_mv, labels_pl)
            loss = 1 * loss_mv + loss_pc

            tf.summary.scalar('loss', loss)

            correct_pc = tf.equal(tf.argmax(pred_pc, 1), tf.to_int64(labels_pl))
            accuracy_pc = tf.reduce_sum(tf.cast(correct_pc, tf.float32)) / float(BATCH_SIZE)

            correct_mv = tf.equal(tf.argmax(pred_mv, 1), tf.to_int64(labels_pl))
            accuracy_mv = tf.reduce_sum(tf.cast(correct_mv, tf.float32)) / float(BATCH_SIZE)

            tf.summary.scalar('accuracy_pc', accuracy_pc)

            # Get params to be updated
            tvars = tf.trainable_variables()
            update_vars = [var for var in tvars if 'conv' not in var.name]
            print (update_vars)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            # tf.summary.scalar('learning_rate', learning_rate_pc)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch, var_list=update_vars)


            # optimizer_mv = tf.train.AdamOptimizer(learning_rate=0.0001)
            # train_op_mv = optimizer_mv.minimize(loss_mv, global_step=batch_mv, var_list=update_vars)

            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            load_var1 = [var for var in tvars if 'conv' in var.name]
            load_var2 = [var for var in tvars if 'fc6' in var.name]
            load_var3 = [var for var in tvars if 'fc7' in var.name]
            load_var4 = [var for var in tvars if 'fc8' in var.name]
            load_var = load_var1 + load_var2 + load_var3 + load_var4
            log_string('-----------------')
            print (load_var)
            saver1 = tf.train.Saver(var_list=load_var)
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables and load pre-trained mvcnn
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})
        saver1.restore(sess, save_path=LOAD_DIR)

        ops = {'pointclouds_pl': pointclouds_pl,
               'images_pl': images_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred_pc': pred_pc,
               'pred_mv': pred_mv,
               'loss_pc': loss_pc,
               'loss_mv': loss_mv,
               'train_op': train_op,
               # 'train_op_mv': train_op_mv,
               'merged': merged,
               'step': batch,}
               # 'step_mv': batch_mv}


        eval_best_acc_pc = 0
        eval_best_acc_mv = 0
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer, dataset_train)
            eval_acc_pc, eval_acc_mv = eval_one_epoch(sess, ops, test_writer, dataset_test)
            if eval_acc_pc >= eval_best_acc_pc:
                eval_best_acc_pc = eval_acc_pc
            if eval_acc_mv >= eval_best_acc_mv:
                eval_best_acc_mv = eval_acc_mv
            log_string('pc best eval accuracy: %f' % eval_best_acc_pc)
            log_string('mv best eval accuracy: %f' % eval_best_acc_mv)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(SAVE_DIR, str(epoch)+"_model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer, dataset_train):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    num_data, num_batches = dataset_train.get_len()

    # # Shuffle train files
    # train_file_idxs = np.arange(0, len(TRAIN_FILES))
    # np.random.shuffle(train_file_idxs)
    # log_string('----' + str(fn) + '-----')

    # current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
    # current_data = current_data[:,0:NUM_POINT,:]
    # current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
    # current_label = np.squeeze(current_label)

    # file_size = current_data.shape[0]
    # num_batches = file_size // BATCH_SIZE

    total_correct_pc = 0
    total_correct_mv = 0
    loss_sum_pc = 0
    loss_sum_mv = 0
    total_seen = 0

    for batch_idx in range(num_batches):
        # start_idx = batch_idx * BATCH_SIZE
        # end_idx = (batch_idx+1) * BATCH_SIZE

        imgs, pcs, lbls = dataset_train.get_batch(idx=batch_idx)

        # Augment batched point clouds by rotation and jittering
        rotated_data = data_util.rotate_point_cloud(pcs)
        jittered_data = data_util.jitter_point_cloud(rotated_data)
        jittered_data = data_util.random_scale_point_cloud(jittered_data)
        jittered_data = data_util.rotate_perturbation_point_cloud(jittered_data)
        jittered_data = data_util.shift_point_cloud(jittered_data)

        feed_dict = {ops['pointclouds_pl']: jittered_data,
                     ops['images_pl']: imgs,
                     ops['labels_pl']: lbls,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val_pc, pred_val_pc, loss_val_mv, pred_val_mv = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss_pc'], ops['pred_pc'], ops['loss_mv'], ops['pred_mv']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val_pc = np.argmax(pred_val_pc, 1)
        correct_pc = np.sum(pred_val_pc == lbls)
        total_correct_pc += correct_pc
        total_seen += BATCH_SIZE
        loss_sum_pc += (loss_val_pc*BATCH_SIZE)

        pred_val_mv = np.argmax(pred_val_mv, 1)
        correct_mv = np.sum(pred_val_mv == lbls)
        total_correct_mv += correct_mv
        loss_sum_mv += (loss_val_mv*BATCH_SIZE)

        # feed_dict = {ops['images_pl']: imgs,
        #              ops['labels_pl']: lbls,
        #              ops['is_training_pl']: is_training,}
        # summary_mv, step_mv, _, loss_val_mv, pred_val_mv = sess.run([ops['merged'], ops['step_mv'],
        #     ops['train_op_mv'], ops['loss_mv'], ops['pred_mv']], feed_dict=feed_dict)
        # train_writer.add_summary(summary_mv, step_mv)
        # pred_val_mv = np.argmax(pred_val_mv, 1)
        # correct_mv = np.sum(pred_val_mv == lbls)
        # total_correct_mv += correct_mv
        # loss_sum_mv += (loss_val_mv*BATCH_SIZE)

        if batch_idx % 25 == 0 and batch_idx != 0:
            log_string('----' + str(batch_idx) + '-----')
            log_string('pc mean loss: %f' % (loss_sum_pc / float(total_seen)))
            log_string('pc accuracy: %f' % (total_correct_pc / float(total_seen)))
            log_string('mv mean loss: %f' % (loss_sum_mv / float(total_seen)))
            log_string('mv accuracy: %f' % (total_correct_mv / float(total_seen)))


        


        
def eval_one_epoch(sess, ops, test_writer, dataset_test):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct_pc = 0
    total_correct_mv = 0
    total_seen = 0
    loss_sum_pc = 0
    loss_sum_mv = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    num_data, num_batches = dataset_test.get_len()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        imgs, pcs, lbls = dataset_test.get_batch(idx=batch_idx)

        feed_dict = {ops['pointclouds_pl']: pcs,
                     ops['images_pl']: imgs,
                     ops['labels_pl']: lbls,
                     ops['is_training_pl']: is_training}
        summary, step, _, loss_val_pc, pred_val_pc, loss_val_mv, pred_val_mv = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss_pc'], ops['pred_pc'], ops['loss_mv'], ops['pred_mv']], feed_dict=feed_dict)
        pred_val_pc = np.argmax(pred_val_pc, 1)
        correct_pc = np.sum(pred_val_pc == lbls)
        total_correct_pc += correct_pc
        total_seen += BATCH_SIZE
        loss_sum_pc += (loss_val_pc*BATCH_SIZE)
        for i in range(BATCH_SIZE):
            l = lbls[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val_pc[i] == l)

        pred_val_mv = np.argmax(pred_val_mv, 1)
        correct_mv = np.sum(pred_val_mv == lbls)
        total_correct_mv += correct_mv
        loss_sum_mv += (loss_val_mv*BATCH_SIZE)


        # feed_dict = {ops['images_pl']: imgs,
        #              ops['labels_pl']: lbls,
        #              ops['is_training_pl']: is_training}
        # summary_mv, step_mv, loss_val_mv, pred_val_mv = sess.run([ops['merged'], ops['step_mv'],
        #     ops['loss_mv'], ops['pred_mv']], feed_dict=feed_dict)
        # pred_val_mv = np.argmax(pred_val_mv, 1)
        # correct_mv = np.sum(pred_val_mv == lbls)
        # total_correct_mv += correct_mv
        # loss_sum_mv += (loss_val_mv*BATCH_SIZE)

    log_string('----------------')
    log_string('pc eval mean loss: %f' % (loss_sum_pc / float(total_seen)))
    eval_acc_pc = (total_correct_pc / float(total_seen))
    log_string('pc eval accuracy: %f'% (eval_acc_pc))
    log_string('pc eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    log_string('mv eval mean loss: %f' % (loss_sum_mv / float(total_seen)))
    eval_acc_mv = (total_correct_mv / float(total_seen))
    log_string('mv eval accuracy: %f'% (eval_acc_mv))

    return eval_acc_pc, eval_acc_mv


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
