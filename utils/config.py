import sys
sys.path.append('../')
import os
import os.path as osp
import configparser

class config(object):
    def __init__(self, cfg_file='config/config.cfg'):
        super(config, self).__init__()
        cfg = configparser.ConfigParser()
        cfg.read(cfg_file)

        # default
        self.data_root = cfg.get('DEFAULT', 'data_root')
        self.data_3d_root = cfg.get('DEFAULT', 'data_3d_root')
        self.data_views_root = cfg.get('DEFAULT', 'data_views_root')
        self.data_points_root = cfg.get('DEFAULT', 'data_points_root')
        self.result_root = cfg.get('DEFAULT', 'result_root')
        self.class_num = cfg.getint('DEFAULT', 'class_num')

        self.ps_each_file = cfg.getint('DEFAULT', 'ps_each_file')
        self.ps_input_num = cfg.getint('DEFAULT', 'ps_input_num')
        self.vis_pc = cfg.getboolean('DEFAULT', 'vis_pc')

        self.model_type = cfg.get('DEFAULT', 'model_type')
        self.views_num = cfg.getint('DEFAULT', 'views_num')
        self.img_size = cfg.getint('DEFAULT', 'img_size')

        # train
        self.cuda = cfg.getboolean('TRAIN', 'cuda')

        self.result_sub_folder = cfg.get('TRAIN', 'result_sub_folder')
        self.ckpt_folder = cfg.get('TRAIN', 'ckpt_folder')
        self.split_folder = cfg.get('TRAIN', 'split_folder')

        self.split_train = cfg.get('TRAIN', 'split_train')
        self.split_test = cfg.get('TRAIN', 'split_test')
        self.ckpt_model = cfg.get('TRAIN', 'ckpt_model')
        self.ckpt_optim = cfg.get('TRAIN', 'ckpt_optim')
        self.ckpt_view_model = cfg.get('TRAIN', 'ckpt_view_model')
        self.log_dir = cfg.get('TRAIN', 'log_dir')

        self.gpu = cfg.get('TRAIN', 'gpu')
        self.model = cfg.get('TRAIN', 'model')
        self.batch_size = cfg.getint('TRAIN', 'batch_size')
        self.max_epoch = cfg.getint('TRAIN', 'max_epoch')
        self.lr = cfg.getfloat('TRAIN', 'lr')
        self.momentum = cfg.getfloat('TRAIN', 'momentum')
        self.optimizer = cfg.get('TRAIN', 'optimizer')
        self.decay_step = cfg.getint('TRAIN', 'decay_step')
        self.decay_rate = cfg.getfloat('TRAIN', 'decay_rate')

        self.check_dirs()

    def check_dir(self, folder):
        if not osp.exists(folder):
            os.mkdir(folder)

    def check_dirs(self):
        self.check_dir(self.data_views_root)
        self.check_dir(self.data_points_root)
        self.check_dir(self.result_root)
        self.check_dir(self.result_sub_folder)
        self.check_dir(self.ckpt_folder)
        self.check_dir(self.split_folder)
        self.check_dir(self.log_dir)