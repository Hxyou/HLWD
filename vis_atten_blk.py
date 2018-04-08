import os
import os.path as osp
import glob
from random import shuffle
from math import isnan
import numpy as np
from tqdm import tqdm
import dataset
from dataset import provider
from utils import config
import matplotlib.pyplot as plt
import bisect
from mpl_toolkits.mplot3d import Axes3D
from utils import heat_map_fun
import h5py as h5


def draw_pc(pc, show=True, save_dir=None):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    if show:
        plt.show()
    if save_dir is not None:
        plt.savefig(save_dir)


class vis_att(object):
    def __init__(self):
        super(vis_att, self).__init__()
        self.cfg = config.config()
        self.shapes = provider.pc_data(self.cfg, state='test', batch_size=1)
        self.shape_list = self.shapes.shape_list
        self.att_mask = h5.File(self.cfg.mask_file, 'r')

    def vis_one_class(self, class_name):
        idxs, names =  self.shapes.get_sp_idxs(class_name)
        for idx in idxs:
            self.vis_one_idx(idx)

    def vis_one_shape(self, shape_name):
        idx, _ = self.shapes.get_sp_idxs(shape_name)
        idx = idx[0]
        self.vis_one_idx(idx)

    def vis_one_idx(self, idx):
        pc, lbl, shape_name = self.shapes[idx]
        self.vis(pc, self.att_mask['mask1'][idx], self.att_mask['mask2'][idx], shape_name)

    def vis(self, pc, mask1, mask2, shape_name):
        for idx in tqdm(range(mask1.shape[1])):
            save_name = osp.join(self.cfg.vis_attn_folder, '%s_mask1_%d.jpg'%(shape_name, idx))
            # save_name = osp.join('/home/fyf/data/mm 2018 data', '%s_mask1_%d.jpg'%(shape_name, idx))
            self._vis(pc, mask1[:, idx], save_name)
        for idx in tqdm(range(mask2.shape[1])):
            save_name = osp.join(self.cfg.vis_attn_folder, '%s_mask2_%d.jpg'%(shape_name, idx))
            # save_name = osp.join('/home/fyf/data/mm 2018 data', '%s_mask2_%d.jpg'%(shape_name, idx))
            self._vis(pc, mask2[:, idx], save_name)

    def vis_sp(self,shape_name, mask_name, sub_idx):
        idx, _ = self.shapes.get_sp_idxs(shape_name)
        idx = idx[0]
        pc, lbl, shape_name = self.shapes[idx]
        if mask_name == 'mask1':
            # save_name = osp.join(self.cfg.vis_attn_folder, '%s_mask1_%d.jpg' % (shape_name, sub_idx))
            save_name = osp.join('/home/fyf/data/mm 2018 data/atten', '%s_mask1_%d.jpg'%(shape_name, sub_idx))
            mask = self.att_mask['mask1'][idx, :, sub_idx]
        else:
            # save_name = osp.join(self.cfg.vis_attn_folder, '%s_mask2_%d.jpg' % (shape_name, sub_idx))
            save_name = osp.join('/home/fyf/data/mm 2018 data/atten', '%s_mask2_%d.jpg'%(shape_name, sub_idx))
            mask = self.att_mask['mask2'][idx, :, sub_idx]

        self._vis(pc, mask, save_name, show=False)

    def _vis(self, pc, c, save_name, show=False):
        # color = heat_map_fun.get_heatmap_from_prob(c)
        color = heat_map_fun.get_norm_heatmap_from_prob(c)
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=color)
        l_min = min(pc[:, 0].min(), pc[:, 1].min(), pc[:, 2].min())
        l_max = min(pc[:, 0].max(), pc[:, 1].max(), pc[:, 2].max())
        l_min = min(-abs(l_min), -abs(l_max))
        l_max = min(abs(l_min), abs(l_max))
        ax.set_xlim(l_min, l_max)
        ax.set_ylim(l_min, l_max)
        ax.set_zlim(pc[:, 2].min(), pc[:, 2].max())
        ax.axis('off')
        # ax.set_zlim(-1, 1)
        if show:
            plt.show()
        plt.savefig(save_name)

if __name__ == '__main__':
    v = vis_att()
    # v.vis_one_idx(1)
    # v.vis_one_shape('bed_0543')
    # v.vis_one_shape('monitor_0483')
    # v.vis_one_shape('car_0202')
    v.vis_sp('bottle_0348', 'mask1', 66)