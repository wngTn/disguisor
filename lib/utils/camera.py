import os.path as osp
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import torch

IMAGE_H, IMAGE_W = 1536, 2048

# invert the camera parameters for the reprojection into 3D
def calculate_reprojection_params(params):
    # extrinsic parameters: R = rotation matrix, T = translation vector
    # intrinsic parameters: f = focal length, c = principal point,
    # k, p = distortion coefficients
    R, T, f, c, k, p = unfold_camera_param(params)
    K = np.array([f[0][0], 0, c[0][0], 0, f[1][0], c[1][0], 0, 0, 1])

    K = K.reshape(3, 3)
    K_inv = np.linalg.inv(K)
    R_t = R.T
    T_inv = (-R_t @ T).T
    return K_inv, R_t, T_inv


# reproject one pixel into the corresponding 3D point
def reproject_pixel_in_3D(camera_params, px_coords, depth_mask):
    # convert the pixel into homogeneous coordinates
    homo_px_coords = np.hstack((px_coords, [1]))

    # depth_mask is flipped
    # a pixel (x,y) in the color image can be accessed by (y,x) in the depth mask
    depth = depth_mask[int(px_coords[1])][int(px_coords[0])] / 1000
    # the field of view of the depth camera is smaller than the one for the rgb images
    # need to check whether we have a measurement for the given pixel
    if depth == 0.0:
        return None

    # invert the camera parameters to get a reprojection from 2D into 3D
    K_inv, R_t, T_inv = calculate_reprojection_params(camera_params)
    px_to_depth_cam = K_inv @ homo_px_coords * depth
    depth_cam_to_world = R_t @ px_to_depth_cam + T_inv

    return depth_cam_to_world.reshape(3)


def unfold_camera_param(camera):
    world2color = np.linalg.inv(camera["color2world"])
    R, T = homogenous_to_rot_trans(world2color)
    fx = camera['fx']
    fy = camera['fy']
    # f = 0.5 * (camera['fx'] + camera['fy'])
    f = np.array([[fx], [fy]]).reshape(-1, 1)
    c = np.array([[camera['cx']], [camera['cy']]]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


def rot_trans_to_homogenous(rot, trans):
    """
    Args
        rot: 3x3 rotation matrix
        trans: 3x1 translation vector
    Returns
        4x4 homogenous matrix
    """
    X = np.zeros((4, 4))
    X[:3, :3] = rot
    X[:3, 3] = trans.T
    X[3, 3] = 1
    return X


def homogenous_to_rot_trans(X):
    """
    Args
        x: 4x4 homogenous matrix
    Returns
        rotation, translation: 3x3 rotation matrix, 3x1 translation vector
    """

    return X[:3, :3], X[:3, 3].reshape(3, 1)


def rotation_to_homogenous(vec):
    rot_mat = Rotation.from_rotvec(vec)
    swap = np.identity(4)
    swap = np.zeros((4, 4))
    swap[:3, :3] = rot_mat.as_matrix()
    swap[3, 3] = 1
    return swap

def pt3d_camera_params(camera_params):
    """
    Returns the camera parameters in the form needed by pytorch3d

    :param camera_params: The camera parameters
    :return: The camera parameters in pytorch3d form
    """

    Rs = torch.empty(0)
    Ts = torch.empty(0)
    fls = torch.empty(0)
    pps = torch.empty(0)
    image_sizes = []

    for params in camera_params:

        fx = params['fx']
        fy = params['fy']
        cx = params['cx']
        cy = params['cy']

        world2color = np.zeros((4, 4))
        world2color[:3, :3] = params['R']
        world2color[:3, 3] = params['T'].flatten()

        F = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        world2color = F @ world2color

        R = torch.tensor(np.array([world2color[:3, :3].T]), dtype=torch.float32)
        T = torch.tensor(np.array([world2color[:3, 3]]), dtype=torch.float32)

        focal_length = torch.tensor([[fx, fy]], dtype=torch.float32)

        principal_point = torch.tensor([[cx, cy]], dtype=torch.float32)

        Rs = torch.cat((Rs, R), 0)
        Ts = torch.cat((Ts, T), 0)
        fls = torch.cat((fls, focal_length), 0)
        pps = torch.cat((pps, principal_point), 0)
        image_sizes.append([IMAGE_H, IMAGE_W])

    return Rs, Ts, fls, pps, image_sizes

def load_camera_params(dataset_root):
    """Loads the parameters of all the cameras in the dataset_root directory

    Args:
        dataset_root (string): the path to the camera data

    Returns:
        List[dict]: a list of dicts where the parameters are in
    """
    scaling = 1000
    cameras = list(sorted(next(os.walk(dataset_root))[1]))
    camera_params = []
    for cam in cameras:
        ds = {"id": cam}
        intrinsics = osp.join(dataset_root, cam, 'camera_calibration.yml')
        assert osp.exists(intrinsics)
        fs = cv2.FileStorage(intrinsics, cv2.FILE_STORAGE_READ)
        color_intrinsics = fs.getNode("undistorted_color_camera_matrix").mat()
        ds['fx'] = color_intrinsics[0, 0]
        ds['fy'] = color_intrinsics[1, 1]
        ds['cx'] = color_intrinsics[0, 2]
        ds['cy'] = color_intrinsics[1, 2]

        # distortion parameters can be neglected
        dist = fs.getNode("color_distortion_coefficients").mat()
        ds['k'] = np.array(dist[[0, 1, 4, 5, 6, 7]])
        ds['p'] = np.array(dist[2:4])
        # ds['k'] = np.zeros((3, 1))
        # ds['p'] = np.zeros((2, 1))

        depth2color_r = fs.getNode('depth2color_rotation').mat()
        # depth2color_t is in mm by default, change all to meters
        depth2color_t = fs.getNode('depth2color_translation').mat() / scaling

        depth2color = rot_trans_to_homogenous(depth2color_r,
                                              depth2color_t.reshape(3))
        ds["depth2color"] = depth2color

        extrinsics = osp.join(dataset_root, cam, "world2camera.json")
        assert osp.exists(extrinsics)
        with open(extrinsics, 'r') as f:
            ext = json.load(f)
            trans = np.array([x for x in ext['translation'].values()])

            _R = ext['rotation']
            rot = Rotation.from_quat([_R['x'], _R['y'], _R['z'],
                                      _R['w']]).as_matrix()
            ext_homo = rot_trans_to_homogenous(rot, trans)

        # flip coordinate transform back to opencv convention
        yz_flip = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
        YZ_SWAP = rotation_to_homogenous(np.pi / 2 * np.array([1, 0, 0]))

        # first swap into OPENGL convention, then we can apply intrinsics.
        # then swap into our own Z-up prefered format..
        depth2world = YZ_SWAP @ ext_homo @ yz_flip

        ds["depth2world"] = depth2world
        color2world = depth2world @ np.linalg.inv(depth2color)

        ds["color2world"] = color2world

        world2color = np.linalg.inv(color2world)
        ds["world2color"] = world2color
        R, T = homogenous_to_rot_trans(world2color)
        ds["R"] = R
        ds["T"] = T

        camera_params.append(ds)

    return camera_params


def project_points_radial(x, R, T, K, k, p):
    """
    Args
        x: Nx3 points in world coordinates R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        K: 3x3 Camera intrinsic matrix
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    x = x
    # x = np.multiply([-1, 1, 1], x)
    # world2camera
    # https://www-users.cs.umn.edu/~hspark/CSci5980/Lec2_ProjectionMatrix.pdf
    xcam = R.dot(x.T) + T
    xcam = K @ xcam

    # perspective projection to map into pixels:
    # divide by the third component which represents the depth
    ypixel = xcam[:2] / (xcam[2] + 1e-5)

    return ypixel.T


def project_points_opencv(x, R, T, K, k, p):
    dist_coefs = np.concatenate([k[0:2].T[0], p.T[0], k[2:].T[0]])
    # rvec, T perform a change of basis from world to camera coordinate system
    rvec = cv2.Rodrigues(R)[0]
    # project from 3D to 2D. projectPoints handles rotation and translation
    points_2d = cv2.projectPoints(x, rvec, T, K, dist_coefs)
    # TODO: why does projectPoints nest arrays like this?
    return np.array([x[0] for x in points_2d[0]])


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)

    K = np.array([f[0][0], 0, c[0][0], 0, f[1][0], c[1][0], 0, 0, 1])
    K = K.reshape(3, 3)
    # loc2d_opencv = project_points_opencv(x, R, T, K, k, p)
    loc2d = project_points_radial(x, R, T, K, k, p)

    # print(camera["id"])
    # print("------------")
    # print(f" loc2d -> {loc2d} \n opencv -> {loc2d_opencv}")
    return loc2d
