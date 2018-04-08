import utils.config
import os
import sys
sys.path.append('../')
import os.path as osp
import pickle
import glob


def split_views():
    cfg = utils.config.config()
    train_view = get_filenames('train', cfg.data_views_root, 'view')
    test_view = get_filenames('test', cfg.data_views_root, 'view')

    train = [v for k, v in train_view.items()]
    test = [v for k, v in test_view.items()]

    print('train num: %d'% len(train))
    print('test num: %d'% len(test))

    with open(cfg.split_train, 'wb') as f:
        pickle.dump(train, f)
    with open(cfg.split_test, 'wb') as f:
        pickle.dump(test, f)


def split_pc_views():
    cfg = utils.config.config()
    train = []
    test = []

    train_view = get_filenames('train', cfg.data_views_root, 'view')
    test_view = get_filenames('test', cfg.data_views_root, 'view')

    train_pc = get_filenames('train', cfg.data_points_root, 'pc')
    test_pc = get_filenames('test', cfg.data_points_root, 'pc')

    for shape_name in train_view.keys():
        train.append({'label': train_view[shape_name]['label'],
                      'label_name': train_view[shape_name]['label_name'],
                      'imgs': train_view[shape_name]['imgs'],
                      'pc': train_pc[shape_name]['pc'],
                      'shape_name': shape_name})

    for shape_name in test_view.keys():
        test.append({'label': test_view[shape_name]['label'],
                      'label_name': test_view[shape_name]['label_name'],
                      'imgs': test_view[shape_name]['imgs'],
                      'pc': test_pc[shape_name]['pc'],
                      'shape_name': shape_name})

    test = sorted(test, key=lambda x: x['shape_name'])
    train = sorted(train, key=lambda x: x['shape_name'])

    print('train num: %d' % len(train))
    print('test num: %d' % len(test))

    with open(cfg.split_train, 'wb') as f:
        pickle.dump(train, f)
    with open(cfg.split_test, 'wb') as f:
        pickle.dump(test, f)


def get_one_class_view_list(d_root, lbl, lbl_idx):
    """
    get all structed filenames in one class
    :param d_root:
    :param data_views:
    :return:
    -bench_0001--
               |-/home/fyf/code/mvcnn/data/12_ModelNet40/bench/train/bench_0001_001.jpg
               |-/home/fyf/code/mvcnn/data/12_ModelNet40/bench/train/bench_0001_002.jpg
               |-......
    -bench_0002--
               |-......
    """
    full_names = glob.glob(osp.join(d_root, '*.jpg'))
    full_names = sorted(full_names)
    raw_structed_data = {}
    structed_data = {}
    names = [osp.split(name)[1] for name in full_names]
    for _idx, name in enumerate(names):
        shape_name = name[:name.rfind('_')]
        if shape_name not in raw_structed_data:
            raw_structed_data[shape_name] = [full_names[_idx]]
        else:
            raw_structed_data[shape_name].append(full_names[_idx])
    for k, v in raw_structed_data.items():
        structed_data[k] = {'label': lbl_idx,
                            'label_name': lbl,
                            'imgs': v,
                            'shape_name': k}

    return structed_data


def get_one_class_pc_list(d_root, lbl, lbl_idx):
    full_names = glob.glob(osp.join(d_root, '*.npy'))
    full_names = sorted(full_names)
    raw_structed_data = {}
    structed_data = {}
    names = [osp.split(name)[1] for name in full_names]
    for _idx, name in enumerate(names):
        shape_name = name[:name.rfind('.')]
        raw_structed_data[shape_name] = full_names[_idx]

    for k, v in raw_structed_data.items():
        structed_data[k] = {'label': lbl_idx,
                            'label_name': lbl,
                            'pc': v,
                            'shape_name': k}

    return structed_data


def get_filenames(data_state, root, data_type):
    filenames = {}
    data_all = glob.glob(osp.join(root, '*'))
    data_all = sorted(data_all)
    data_all = [data for data in data_all if osp.isdir(data)]
    for _idx, d in enumerate(data_all):
        d_lbl = osp.split(d)[1]
        d_lbl_idx = _idx
        d_root = osp.join(root, d_lbl, data_state)
        if data_type=='view':
            d_dict = get_one_class_view_list(d_root, d_lbl, d_lbl_idx)
        else:
            d_dict = get_one_class_pc_list(d_root, d_lbl, d_lbl_idx)
        filenames.update(d_dict)
    return filenames


if __name__ == '__main__':
    split_pc_views()