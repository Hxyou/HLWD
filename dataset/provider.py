import sys
sys.path.append('../')
import utils.config
import utils.split_fun
import utils.generate_pc
import os.path as osp
import pickle
import numpy as np
from PIL import Image
import random


class view_data(object):
    def __init__(self, cfg, state='train', batch_size=4, shuffle=False, img_sz=224):
        super(view_data, self).__init__()
        if not osp.exists(cfg.split_train) or not osp.exists(cfg.split_test):
            utils.split_fun.split_views()

        if state=='train':
            with open(cfg.split_train, 'rb') as f:
                self.shape_list = pickle.load(f)
        else:
            with open(cfg.split_test, 'rb') as f:
                self.shape_list = pickle.load(f)

        if shuffle:
            random.shuffle(self.shape_list)

        print('%s data num: %d'%(state, len(self.shape_list)) )

        self.img_sz = img_sz
        self.cfg = cfg
        self.batch_size = batch_size

    def get_batch(self, idx):
        start_idx = idx*self.batch_size
        end_idx = (idx+1)*self.batch_size

        imgs = np.zeros((self.batch_size, 12, self.img_sz, self.img_sz, 3)).astype(np.float32)
        lbls = np.zeros(self.batch_size).astype(np.int64)
        for _idx, shape_idx in enumerate(range(start_idx, end_idx)):
            lbls[_idx] = self.shape_list[shape_idx]['label']
            for view_idx, img_name in enumerate(self.shape_list[shape_idx]['imgs']):
                img = Image.open(img_name)
                img = img.resize((self.img_sz, self.img_sz))
                img = np.array(img)/255.0
                imgs[_idx, view_idx] = img
        return imgs, lbls

    def __len__(self):
        return len(self.shape_list)//self.batch_size

    def get_len(self):
        return len(self.shape_list), len(self.shape_list)//self.batch_size


class pc_view_data(object):
    def __init__(self, cfg, state='train', batch_size=4, shuffle=False, img_sz=227, ps_input_num=1024):
        super(pc_view_data, self).__init__()
        if not osp.exists(cfg.split_train) or not osp.exists(cfg.split_test):
            utils.split_fun.split_pc_views()

        if state=='train':
            with open(cfg.split_train, 'rb') as f:
                self.shape_list = pickle.load(f)
        else:
            with open(cfg.split_test, 'rb') as f:
                self.shape_list = pickle.load(f)

        if shuffle:
            random.shuffle(self.shape_list)

        print('%s data num: %d'%(state, len(self.shape_list)) )

        self.img_sz = img_sz
        self.cfg = cfg
        self.batch_size = batch_size
        self.ps_input_num = ps_input_num

    def get_batch(self, idx):
        start_idx = idx*self.batch_size
        end_idx = (idx+1)*self.batch_size

        imgs = np.zeros((self.batch_size, 12, self.img_sz, self.img_sz, 3)).astype(np.float32)
        pcs = np.zeros((self.batch_size, self.ps_input_num, 3)).astype(np.float32)
        lbls = np.zeros(self.batch_size).astype(np.int64)
        for _idx, shape_idx in enumerate(range(start_idx, end_idx)):
            lbls[_idx] = self.shape_list[shape_idx]['label']
            pcs[_idx] = np.load(self.shape_list[shape_idx]['pc'])[:self.ps_input_num]
            for view_idx, img_name in enumerate(self.shape_list[shape_idx]['imgs']):
                img = Image.open(img_name)
                img = img.resize((self.img_sz, self.img_sz))
                img = np.array(img)/255.0
                imgs[_idx, view_idx] = img

        return imgs, pcs, lbls

    def __len__(self):
        return len(self.shape_list)//self.batch_size

    def get_len(self):
        return len(self.shape_list), len(self.shape_list)//self.batch_size


if __name__ == '__main__':
    cfg = utils.config.config()
    # vd = view_data(cfg, state='test', batch_size=8, shuffle=True)
    # batch_len = len(vd)
    # imgs, lbls = vd.get_batch(307)
    # print(batch_len)
    # print(imgs.shape)
    # print(lbls)
    # Image.fromarray((imgs[0][0]*255).astype(np.uint8)).show()


    pvd = pc_view_data(cfg, state='test', batch_size=8, shuffle=True)
    batch_len = len(pvd)
    imgs, pcs, lbls = pvd.get_batch(307)
    print(batch_len)
    print(imgs.shape)
    print(lbls)
    Image.fromarray((imgs[0][0]*255).astype(np.uint8)).show()
    utils.generate_pc.draw_pc(pcs[0])