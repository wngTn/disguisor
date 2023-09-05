# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from scipy.spatial.transform import Rotation

import cv2


def unfold_camera_param(camera):
    R = camera['R']
    T = camera['T']
    fx = camera['fx']
    fy = camera['fy']
    # f = 0.5 * (camera['fx'] + camera['fy'])
    f = np.array([[fx], [fy]]).reshape(-1, 1)
    c = np.array([[camera['cx']], [camera['cy']]]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p

def project_point_radial(x, R, T, f, c, k, p):
    n = x.shape[0]
    x = x
    K = np.array([f[0][0], 0, c[0][0],
                  0, f[1][0], c[1][0],
                  0, 0, 1])
    K = K.reshape(3,3)
    # world2camera
    # https://www-users.cs.umn.edu/~hspark/CSci5980/Lec2_ProjectionMatrix.pdf
    # weird voxelpose convention.. we return -T because of this
    xcam = R.dot(x.T) + T
    xcam = K @ xcam

    ypixel = xcam[:2] / (xcam[2]+1e-5)
    # print(xcam[2])

    # r2 = np.sum(y**2, axis=0)
    # radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
    #                        np.array([r2, r2**2, r2**3]))
    # tan = p[0] * y[1] + p[1] * y[0]
    # y = y * np.tile(radial + 2 * tan,
    #                 (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    # ypixel = np.multiply(f, y) + c
    return ypixel.T

def _project_point_radial(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """

    #R R R T  X
    #R R R T  Y
    #R R R T  Z
    #0 0 0 1  1
    # World -> Camera
    n = x.shape[0]
    xcam = R.dot(x.T - T)
    y = xcam[:2] / (xcam[2]+1e-5)
    # print(xcam[2])

    r2 = np.sum(y**2, axis=0)
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
                           np.array([r2, r2**2, r2**3]))
    tan = p[0] * y[1] + p[1] * y[0]
    y = y * np.tile(radial + 2 * tan,
                    (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    ypixel = np.multiply(f, y) + c
    return ypixel.T


def project_pose(x, camera, holistic=False):
    R, T, f, c, k, p = unfold_camera_param(camera)
    if holistic:
        return project_point_radial(x, R, T, f, c, k, p)
    else:
        return _project_point_radial(x, R, T, f, c, k, p)


def project_pose_opencv(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)
    cam_mat = np.array([[f[0][0], 0, c[0][0]],
                        [0, f[1][0], c[1][0]],
                        [0, 0, 1]])
    dist_coefs = np.concatenate([k[0:2].T[0], p.T[0], k[2:].T[0]])
    rvec = cv2.Rodrigues(R)[0].T[0]
    # project from 3D to 2D. projectPoints handles rotation and translation
    points_2d = cv2.projectPoints(x, rvec, T, cam_mat, dist_coefs)
    # TODO: why does projectPoints nest arrays like this?
    return np.array([x[0] for x in points_2d[0]])


def world_to_camera_frame(x, R, T):
    """
    Args
        x: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 3d points in camera coordinates
    """

    xcam = R.dot(x.T - T)  # rotate and translate
    return xcam.T


def camera_to_world_frame(x, R, T):
    """
    Args
        x: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 points in world coordinates
    """

    xcam = R.T.dot(x.T) + T  # rotate and translate
    return xcam.T


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
