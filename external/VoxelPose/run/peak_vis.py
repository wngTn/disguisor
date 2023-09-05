"""
Generating some files to see how well the 3D keypoints are
"""
import cv2
import glob
import os
import numpy as np
from pathlib import Path
import json
from collections import OrderedDict
from scipy.spatial.transform import Rotation
from matplotlib import colors as cl

import _init_paths
import utils.cameras_cpu as cam_utils
from utils.cameras_cpu import project_pose
from utils.cameras_cpu import rot_trans_to_homogenous, homogenous_to_rot_trans
from utils.cameras_cpu import rotation_to_homogenous

PATH_TO_IMAGES = Path("/Users/tonywang/Documents/University/Master/disguisor/code/data/Hard")
PATH_TO_OUTPUT = Path("./tmp")
colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']

os.makedirs(str(PATH_TO_OUTPUT), exist_ok=True)

def vis_img2d(data, frame_id, path_to_images_for_trial_and_phase, cameras):
    images = []
    for c, cn in enumerate(['cn01', 'cn02', 'cn03', 'cn04']):
        image_path = f"{path_to_images_for_trial_and_phase}/{cn}/{str(frame_id).zfill(10)}_color.jpg"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        for i in range(len(data)):
            keypoints_3d = np.array(data[i]['keypoints3d'])[:, :3] * 1000
            points2d = cam_utils.project_pose(keypoints_3d, cameras[f"{c}"], True)
            for point2d in points2d:
                image = cv2.circle(image, (int(point2d[0]), int(point2d[1])), 4, tuple(reversed(255 * np.array(cl.to_rgb(colors[int(i % 10)])))), 7)
                        
        images.append(image)
    
    images = [cv2.resize(x, (int(x.shape[1] * 0.3), int(x.shape[0] * 0.3))) for x in images]

    left = np.concatenate([images[0], images[1]], axis=0)
    right = np.concatenate([images[2], images[3]], axis=0)

    combined = np.concatenate([left, right], axis=1)

    cv2.imwrite(str(PATH_TO_OUTPUT / Path(f"{str(frame_id).zfill(4)}.jpg")), combined)
    print("Saved", str(PATH_TO_OUTPUT / Path(f"{str(frame_id).zfill(4)}.jpg")))


def read_json(path_to_json):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
        return data
    
def _get_cams(path_to_images_from_trial_and_phase):
    # bring our calibration files into format of voxelpose
    cameras = OrderedDict()
    cams = sorted(next(os.walk(path_to_images_from_trial_and_phase))[1])
    cams = [x for x in cams if not x.startswith('.')]
    for idx, cam_id in enumerate(cams):
        ds = _get_single_cam(path_to_images_from_trial_and_phase, cam_id)
        cameras[str(int(cam_id[-1]) - 1)] = ds

    for id, cam in cameras.items():
        for k, v in cam.items():
            cameras[id][k] = np.array(v)

    return cameras

def _get_single_cam(path_to_images_from_trial_and_phase, cam):
    ds = OrderedDict()
    intrinsics = os.path.join(path_to_images_from_trial_and_phase, cam, 'camera_calibration.yml')
    ds['id'] = int(cam[-1])
    print(intrinsics)
    assert os.path.exists(intrinsics)
    fs = cv2.FileStorage(intrinsics, cv2.FILE_STORAGE_READ)
    color_intrinsics = fs.getNode("undistorted_color_camera_matrix").mat()
    ds['fx'] = color_intrinsics[0, 0]
    ds['fy'] = color_intrinsics[1, 1]
    ds['cx'] = color_intrinsics[0, 2]
    ds['cy'] = color_intrinsics[1, 2]
    # images are undistorted! Just put 0. Voxelpose assumes just 4 dist coeffs

    ds['k'] = np.zeros((3, 1))
    ds['p'] = np.zeros((2, 1))

    depth2color_r = fs.getNode('depth2color_rotation').mat()
    # depth2color_t is in mm by default, change all to meters
    depth2color_t = fs.getNode('depth2color_translation').mat()

    depth2color = rot_trans_to_homogenous(depth2color_r, depth2color_t.reshape(3))
    ds["depth2color"] = depth2color

    extrinsics = os.path.join(path_to_images_from_trial_and_phase, cam, "world2camera.json")
    with open(extrinsics, 'r') as f:
        ext = json.load(f)
        ext = ext if 'value0' not in ext else ext['value0']
        trans = np.array([x for x in ext['translation'].values()])
        trans = trans * 1000
        _R = ext['rotation']
        rot = Rotation.from_quat([_R['x'], _R['y'], _R['z'], _R['w']]).as_matrix()
        ext_homo = rot_trans_to_homogenous(rot, trans)
        # flip coordinate transform back to opencv convention

    yz_flip = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
    YZ_SWAP = rotation_to_homogenous(np.pi / 2 * np.array([1, 0, 0]))

    depth2world = YZ_SWAP @ ext_homo @ yz_flip
    color2world = depth2world @ np.linalg.inv(depth2color)
    R, T = homogenous_to_rot_trans(np.linalg.inv(color2world))
    ds["R"] = R
    ds["T"] = T
    return ds

def read_keypoints(trial, phase):

    path_to_images_for_trial_and_phase = PATH_TO_IMAGES / Path(trial) / Path(phase)
    path_to_images_for_trial_and_phase = str(path_to_images_for_trial_and_phase)

    path_to_keypoints_for_trial_and_phase = Path(str(PATH_TO_IMAGES) + '_keypoints') / Path(trial) / Path(phase) / Path("keypoints_3d")
    path_to_keypoints_for_trial_and_phase = str(path_to_keypoints_for_trial_and_phase)

    cameras = _get_cams(path_to_images_for_trial_and_phase)

    all_keypoint_files = list(sorted(glob.glob(path_to_keypoints_for_trial_and_phase + '/*.json')))
    all_frame_ids = [int(os.path.basename(x)[:-10]) for x in list(sorted(glob.glob(path_to_images_for_trial_and_phase + '/cn01/*_color.jpg')))]

    assert len(all_keypoint_files) == len(all_frame_ids), f"Different lengths"    

    indices = np.random.choice(len(all_keypoint_files), 10)

    keypoint_files = [all_keypoint_files[i] for i in indices]
    frame_ids = [all_frame_ids[i] for i in indices]

    for i, keypoint_file in enumerate(keypoint_files):
        data = read_json(keypoint_file)
        vis_img2d(data, frame_ids[i], path_to_images_for_trial_and_phase, cameras)


def main():
    read_keypoints("211209_animal_trial_15", "MON")

if __name__=='__main__':
    main()



