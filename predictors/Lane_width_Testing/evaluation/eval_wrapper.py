from predictors.Lane_width_Testing.data.dataloader import get_test_loader
from predictors.Lane_width_Testing.evaluation.tusimple.lane import LaneEval
import os, json, torch, scipy
import numpy as np
import platform

def generate_lines(out, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):
    col_sample = np.linspace(0, shape[1] - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    for j in range(out.shape[0]):
        out_j = out[j].data.cpu().numpy()
        if flip_updown:
            out_j = out_j[:, ::-1, :]
        if localization_type == 'abs':
            out_j = np.argmax(out_j, axis=0)
            out_j[out_j == griding_num] = -1
            out_j = out_j + 1
        elif localization_type == 'rel':
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == griding_num] = 0
            out_j = loc
        else:
            raise NotImplementedError
        name = names[j]

        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            fp.write(
                                '%d %d ' % (int(out_j[k, i] * col_sample_w * 1640 / 800) - 1, int(590 - k * 20) - 1))
                    fp.write('\n')

def run_test(net, data_root, exp_name, work_dir, griding_num, use_aux, batch_size=8):
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    loader = get_test_loader(batch_size, data_root, 'CULane')
    for i, data in enumerate(loader):
        imgs, names = data
        with torch.no_grad():
            out = net(imgs)
        if len(out) == 2 and use_aux:
            out, seg_out = out

        generate_lines(out, imgs[0, 0].shape, names, output_path, griding_num, localization_type='rel', flip_updown=True)

def generate_tusimple_lines(out, shape, griding_num, localization_type='rel'):
    out = out.data.cpu().numpy()
    out_loc = np.argmax(out, axis=0)

    if localization_type == 'rel':
        prob = scipy.special.softmax(out[:-1, :, :], axis=0)
        idx = np.arange(griding_num)
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        loc[out_loc == griding_num] = griding_num
        out_loc = loc
    lanes = []
    for i in range(out_loc.shape[1]):
        out_i = out_loc[:, i]
        lane = [int(round((loc + 0.5) * 1280.0 / (griding_num - 1))) if loc != griding_num else -2 for loc in out_i]
        lanes.append(lane)
    return lanes

def run_test_tusimple(net, data_root, work_dir, exp_name, griding_num, use_aux, batch_size=8):
    output_path = os.path.join(work_dir, exp_name + '.txt')
    with open(output_path, 'w') as fp:
        loader = get_test_loader(batch_size, data_root, 'Tusimple')
        for i, data in enumerate(loader):
            imgs, names = data
            with torch.no_grad():
                out = net(imgs)
            if len(out) == 2 and use_aux:
                out = out[0]
            for i, name in enumerate(names):
                tmp_dict = {
                    'lanes': generate_tusimple_lines(out[i], imgs[0, 0].shape, griding_num),
                    'h_samples': [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
                                  270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
                                  430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580,
                                  590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710],
                    'raw_file': name,
                    'run_time': 10
                }
                json_str = json.dumps(tmp_dict)
                fp.write(json_str + '\n')

def eval_lane(net, dataset, data_root, work_dir, griding_num, use_aux, exp_name):
    net.eval()
    if dataset == 'CULane':
        run_test(net, data_root, exp_name, work_dir, griding_num, use_aux)
    elif dataset == 'Tusimple':
        run_test_tusimple(net, data_root, work_dir, exp_name, griding_num, use_aux)

def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k: v for k, v in zip(keys, values)}
    return res

def call_culane_eval(data_dir, exp_name, output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir = os.path.join(output_path, exp_name) + '/'

    w_lane = 30
    iou = 0.5
    im_w = 1640
    im_h = 590
    frame = 1
    list_files = [
        'test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt',
        'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt'
    ]

    eval_cmd = './evaluation/culane/evaluate'
    if platform.system() == 'Windows':
        eval_cmd = eval_cmd.replace('/', os.sep)

    for i, test_file in enumerate(list_files):
        list_path = os.path.join(data_dir, 'list/test_split', test_file)
        out_path = os.path.join(output_path, 'txt', f'out{i}_{test_file.split("_")[1].split(".")[0]}.txt')
        os.system(f'{eval_cmd} -a {data_dir} -d {detect_dir} -i {data_dir} -l {list_path} '
                  f'-w {w_lane} -t {iou} -c {im_w} -r {im_h} -f {frame} -o {out_path}')
