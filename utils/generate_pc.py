import os
import os.path as osp
import glob
from random import shuffle
from math import isnan
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import bisect
from mpl_toolkits.mplot3d import Axes3D


def des(a, b):
    return np.linalg.norm(a - b)


def get_info(shape_dir):
    splits = shape_dir.split('/')
    class_name = splits[-3]
    set_name = splits[-2]
    file_name = splits[-1].split('.')[0]
    return class_name, set_name, file_name


def random_point_triangle(a, b, c):
    r1 = np.random.random()
    r2 = np.random.random()
    p = np.sqrt(r1) * (r2 * c + b * (1-r2)) + a * (1-np.sqrt(r1))
    return p


def triangle_area(p1, p2, p3):
    a = des(p1, p2)
    b = des(p1, p3)
    c = des(p2, p3)
    p = (a+b+c)/2.0
    area = np.sqrt(p*(p-a)*(p-b)*(p-c))
    if isnan(area):
        # print('find nan')
        area = 1e-6
    return area


def uniform_sampling(points, faces, n_samples):
    sampled_points = []
    total_area = 0
    cum_sum = []
    for _idx, face in enumerate(faces):
        total_area += triangle_area(points[face[0]], points[face[1]], points[face[2]])
        if isnan(total_area):
            print('find nan')
        cum_sum.append(total_area)

    for _idx in range(n_samples):
        tmp = np.random.random()*total_area
        face_idx = bisect.bisect_left(cum_sum, tmp)
        pc = random_point_triangle(points[faces[face_idx][0]],
                                   points[faces[face_idx][1]],
                                   points[faces[face_idx][2]])
        sampled_points.append(pc)
    return np.array(sampled_points)


def normal_pc(pc, L):
    """
    normalize point cloud in range L
    :param pc: type list
    :param L:
    :return: type list
    """
    pc_new = []
    pc_L_max = np.sqrt(np.sum(pc ** 2, 1)).max()
    return pc/pc_L_max*L

def get_pc(shape, point_each):
    points = []
    faces = []
    with open(shape, 'r') as f:
        line = f.readline().strip()
        if line == 'OFF':
            num_verts, num_faces, num_edge = f.readline().split()
            num_verts = int(num_verts)
            num_faces = int(num_faces)
        else:
            num_verts, num_faces, num_edge = line[3:].split()
            num_verts = int(num_verts)
            num_faces = int(num_faces)

        for idx in range(num_verts):
            line = f.readline()
            point = [float(v) for v in line.split()]
            points.append(point)

        for idx in range(num_faces):
            line = f.readline()
            face = [int(t_f) for t_f in line.split()]
            faces.append(face[1:])

    points = np.array(points)
    pc = normal_pc(points, 10)
    pc = uniform_sampling(pc, faces, point_each)

    pc = normal_pc(pc, 1)

    return pc


def generate(cfg):
    shape_all = glob.glob(osp.join(cfg.data_3d_root, '*', '*', '*.off'))
    # shape_all = sorted(shape_all)
    for shape in tqdm(shape_all):
        # if shape.find('bed_0039') == -1:
        #     continue
        class_name, set_name, file_name = get_info(shape)
        new_folder = osp.join(cfg.data_points_root, class_name, set_name)
        new_dir = osp.join(new_folder, file_name)
        if osp.exists(new_dir+'.npy'):
            if cfg.vis_pc and not osp.exists(new_dir+'.jpg'):
                pc = np.load(new_dir+'.npy')
                draw_pc(pc, show=False, save_dir=new_dir+'.jpg')
        else:
            pc = get_pc(shape, cfg.ps_each_file)
            if not osp.exists(new_folder):
                os.makedirs(new_folder)
            np.save(new_dir+'.npy', pc)
            if cfg.vis_pc:
                draw_pc(pc, show=False, save_dir=new_dir+'.jpg')


def draw_pc(pc, show=True, save_dir=None):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    if show:
        plt.show()
    if save_dir is not None:
        plt.savefig(save_dir)

if __name__ == '__main__':
    file_name = '/home/fyf/code/data/pc_ModelNet40/bench/train/bench_0151.npy'
    pc = np.load(file_name)
    draw_pc(pc)


