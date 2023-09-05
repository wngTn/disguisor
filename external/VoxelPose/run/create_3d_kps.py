import math
import numpy as np
import torchvision
import cv2
import os
import matplotlib
from pathlib import Path

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as cl
import argparse

import _init_paths
from core.config import update_config
from core.config import config
from utils.utils import create_logger, load_backbone_panoptic
import torchvision.transforms as transforms
import torch.utils.data
import torch.utils.data.distributed
import torch
import torch.backends.cudnn as cudnn
import dataset
import models
import pickle
import json
import shutil
import time

import utils.cameras as cameras

'''
This file is part of debugging the code to visualize some things
'''


# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize your network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', type=str, required=True)
    parser.add_argument(
        '-v', '--visualize', action='store_true', help='visualize the results')
    parser.add_argument(
        '--pretrained_model', help='the path to the pretrained model',
    )

    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    return args

def image_2d_with_anno(meta, preds, file_name, batch_size):
    images = []
    for m in range(len(meta)):
        image_file = meta[m]['image'][0]
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        meta[m]['camera']['R'] = meta[m]['camera']['R'][0]
        meta[m]['camera']['T'] =  meta[m]['camera']['T'][0]
        meta[m]['camera']['k'] = meta[m]['camera']['k'][0]
        meta[m]['camera']['p'] = meta[m]['camera']['p'][0]

        colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
        for i in range(batch_size):
            if preds is not None:
                pred = preds[i]
                for n in range(len(pred)):
                    joint = pred[n]
                    if joint[0, 3] >= 0:
                        points3d = torch.from_numpy(joint[:, :3])
                        points2d = cameras.project_pose(points3d, meta[m]['camera'], True)
                        for point2d in points2d:
                            image = cv2.circle(image, (int(point2d[0]), int(point2d[1])), 4, tuple(reversed(255 * np.array(cl.to_rgb(colors[int(n % 10)])))), 5)
                            
        images.append(image)

    end_image = cv2.hconcat(images)
    cv2.imwrite(str(file_name), end_image)
    print('Saved', str(file_name))
    

def save_easymocap_output(preds, experiment_name):
    output_dir = Path('..') / 'EasyMocap' / 'data' / experiment_name / 'output' / 'keypoints3d'    
    output_dir.mkdir(parents=True, exist_ok=True)

    for k, v in preds.items():
        frame_num = '{:06d}'.format(k)
        file_name = frame_num + '.json'
        file_path = os.path.join(output_dir, file_name)
        frame = []
        for j, vv in enumerate(v):
            res_temp = {
                'id' : j,
                'keypoints3d' : [
                    [round(c, 3) for c in row] for row in vv.tolist()
                ]
            }
            frame.append(res_temp)
        json_string = json.dumps(frame, indent=4)
        with open(file_path, 'w') as outfile:
            outfile.write(json_string)
            print('Saved:', file_path)

def main():
    args = parse_args()

    experiment_name = "hard"
    out_prefix = Path("output") / experiment_name

    if args.pretrained_model is None:
        MODEL_PATH = os.path.join("output", config.DATASET.TRAIN_DATASET, "multi_person_posenet_50", experiment_name, config.TEST.MODEL_FILE)
    else:
        MODEL_PATH = args.pretrained_model
    assert os.path.exists(MODEL_PATH), "Model path does not exist"

    out_prefix.mkdir(parents=True, exist_ok=True)

    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False, os.path.join(config.DATASET.ROOT, f"{experiment_name}_2d_kps.pkl"),
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config,is_train=False)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = MODEL_PATH
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        print('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    model.eval()
    preds = {}
    with torch.no_grad():
        for l, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(test_loader):
            now = time.time()
            pred, _, _, _, _, _ = model(meta=meta, targets_3d=targets_3d[0], input_heatmaps=input_heatmap)
            pred = pred.detach().cpu().numpy()
            then = time.time()

            print("Find person pose in: {} sec".format(then - now))

            frame_num = l 
            preds[frame_num] = []
            if pred is not None:
                pre = pred[0]
                for n in range(len(pre)):
                    joint = pre[n] # joint of one person
                    if joint[0, 3] >= 0:
                        # converts back to meters
                        pruned_joint = np.concatenate((joint[:, :3] / 1000, joint[:, -1].reshape(17, 1)), axis=1)
                        # joints without the third column
                        preds[frame_num].append(pruned_joint)

            
            if args.visualize:
                image_output_path = out_prefix / "visualization"
                image_output_path.mkdir(parents=True, exist_ok=True)
                image_2d_with_anno(meta, pred, image_output_path / f"{l:06}.jpg", 1)

        # saves the predictions
        # with open(os.path.join(f'{out_prefix}', f'pred_{experiment_name}.pkl'), 'wb') as handle:
        #     pickle.dump(preds, handle)

        # creates the output for easymocap
        save_easymocap_output(preds, experiment_name)


if __name__ == '__main__':
    main()

