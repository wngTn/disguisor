import os
import torch
from pathlib import Path
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)
import cv2
import numpy as np
import copy
from math import degrees, atan2
import trimesh

import utils.indices as indices
from core.blending import blend_faces_into_background
import utils.camera as camera_utils

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def weighted_procrustes(src_points,
                        tgt_points,
                        weights=None,
                        weight_thresh=0.,
                        eps=1e-5,
                        return_transform=False):
    r"""
    Compute rigid transformation from `src_points` to `tgt_points` using weighted SVD.
    Modified from PointDSC.
    :param src_points: torch.Tensor (batch_size, num_corr, 3) or (num_corr, 3)
    :param tgt_points: torch.Tensor (batch_size, num_corr, 3) or (num_corr, 3)
    :param weights: torch.Tensor (batch_size, num_corr) or (num_corr,) (default: None)
    :param weight_thresh: float (default: 0.)
    :param eps: float (default: 1e-5)
    :param return_transform: bool (default: False)
    :return R: torch.Tensor (batch_size, 3, 3) or (3, 3)
    :return t: torch.Tensor (batch_size, 3) or (3,)
    :return transform: torch.Tensor (batch_size, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        tgt_points = tgt_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights_norm = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)

    src_centroid = torch.sum(src_points * weights_norm.unsqueeze(2), dim=1, keepdim=True)
    tgt_centroid = torch.sum(tgt_points * weights_norm.unsqueeze(2), dim=1, keepdim=True)
    src_points_centered = src_points - src_centroid
    tgt_points_centered = tgt_points - tgt_centroid

    W = torch.diag_embed(weights)
    H = src_points_centered.permute(0, 2, 1) @ W @ tgt_points_centered
    U, _, V = torch.svd(H)  # H = USV^T
    Ut, V = U.transpose(1, 2), V
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).type(torch.DoubleTensor).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = tgt_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t

def get_transformation_matrix(source_mesh, target_mesh):
    target_points = np.asarray(target_mesh.vertices)
    source_points = np.asarray(source_mesh.vertices)

    transformation_matrix = weighted_procrustes(torch.tensor(source_points, device=device),
                                                torch.tensor(target_points, device=device),
                                                return_transform=True)

    return transformation_matrix.cpu().numpy()

def get_pt3d_mesh(mesh, texture, with_mask=True):
    """
    Converts the mesh into a Pytorch3D face mesh and transfer it to the device
    """

    trimesh_face_mesh = trimesh.load(os.path.join("input", "geometry", "face.obj"),
                        process=False,
                        maintain_order=True)
    uvs = trimesh_face_mesh.visual.uv

    triangles = mesh.triangles
    vertices = mesh.vertices

    if with_mask:
        masked_mesh = trimesh.load(os.path.join("input", "geometry", "face_with_mask.obj"),
                        process=False,
                        maintain_order=True)
        uvs = masked_mesh.visual.uv
        tf = get_transformation_matrix(trimesh_face_mesh, mesh)
        masked_mesh.apply_transform(tf)

        triangles = masked_mesh.faces
        vertices = masked_mesh.vertices

    verts_uvs = torch.tensor(uvs, dtype=torch.float32)[None, ...]
    faces = torch.tensor(np.array(triangles), dtype=torch.int64)[None, ...]
    verts = torch.tensor(np.array(vertices), dtype=torch.float32)[None, ...]
    texture_image = torch.tensor(texture[None, ...], dtype=torch.float32)

    tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces, maps=texture_image)

    return Meshes(verts=[verts[0]], faces=[faces[0]], textures=tex).to(device)


def image_rendering(mesh, Rs, Ts, fls, pps, image_sizes):

    mesh = mesh.extend(len(image_sizes))

    cameras = PerspectiveCameras(R=Rs,
                                 T=Ts,
                                 focal_length=fls,
                                 principal_point=pps,
                                 image_size=image_sizes,
                                 in_ndc=False)

    raster_settings = RasterizationSettings(image_size=(image_sizes[0][0], image_sizes[0][1]),
                                            blur_radius=0.0,
                                            faces_per_pixel=1,
                                            cull_backfaces=True)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
        )).to(device)

    images = renderer(mesh)

    return images


def convertMeshes2PT3DMeshes(mesh, texture_image, trimesh_m_visual_uv):
    """
    Converts a list of open3d meshes to a list of pytorch3d meshes

    :param meshes: List of open3d meshes
    :param texture_image: The image texture of the mesh
    :param trimesh_m_visual_uv: The UV values of the vertices of the mesh
    """

    verts_uvs = torch.tensor(trimesh_m_visual_uv, dtype=torch.float32)[None, ...]
    faces = torch.tensor(np.array(mesh.triangles), dtype=torch.int64)[None, ...]
    verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)[None, ...]

    # Create a textures object
    tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces, maps=texture_image)

    return Meshes(verts=[verts[0]], faces=[faces[0]], textures=tex).to(device)



def get_visibility_flags(mesh_data, camera_params, frame_id, config):
    """
    Update visibility flags for each person in mesh data.
    """
    for person_data in mesh_data:
        person_data["is_visible"] = np.zeros(len(camera_params), dtype=bool)
    check_for_obstruction(mesh_data, camera_params, frame_id, config)

def load_raw_images(data_dir, camera_params, frame_id):
    """
    Load raw images from the data directory.
    """
    return [
        cv2.imread(str(data_dir / cam["id"] / f"{frame_id:010d}_color.jpg"), cv2.IMREAD_UNCHANGED)
        for cam in camera_params
    ]

def get_face_images_for_all_people(mesh_data, Rs, Ts, fls, pps, image_sizes):
    """
    Get face images for all people in the mesh data.
    """
    face_lists = []
    for person_data in mesh_data:
        pt3d_mesh = get_pt3d_mesh(person_data["face_mesh"], person_data["texture"])
        face_images = image_rendering(pt3d_mesh, Rs, Ts, fls, pps, image_sizes)
        face_list = torch.stack([
            face_images[j] if visible else torch.ones_like(face_images[j])
            for j, visible in enumerate(person_data["is_visible"])
        ])
        face_lists.append(face_list)
    return face_lists

def render_bounding_boxes(images, bboxes_frame):
    """
    Render bounding boxes on images.
    """
    anonymized_bbox_frames = []
    for image, bboxes in zip(images, bboxes_frame):
        image = image.copy()
        for x_min, y_min, width, height in bboxes:
            if x_min is not None:
                image = cv2.rectangle(
                    image, 
                    (int(x_min), int(y_min)),
                    (int(x_min + width), int(y_min + height)),
                    (0, 0, 255),
                    3
                )
        anonymized_bbox_frames.append(image)
    return anonymized_bbox_frames

def render(config, mesh_data, frame_id):
    """
    Renders anonymized images.
    """
    data_dir = Path("data") / config.experiment
    camera_params = camera_utils.load_camera_params(str(data_dir))
    Rs, Ts, fls, pps, image_sizes = camera_utils.pt3d_camera_params(camera_params)

    get_visibility_flags(mesh_data, camera_params, frame_id, config)
    raw_images = load_raw_images(data_dir, camera_params, frame_id)
    face_lists = get_face_images_for_all_people(mesh_data, Rs, Ts, fls, pps, image_sizes)

    images, bboxes_frame = blend_faces_into_background(face_lists, raw_images, num_pixels=250, alpha_value=config.alpha_value)
    anonymized_data = {"anonymized_images": images}

    if config.add_bboxes:
        anonymized_data["anonymized_bbox_images"] = render_bounding_boxes(images, bboxes_frame)
    
    return anonymized_data


def check_for_obstruction(data, camera_params, frame_id, config):
    """
    Checks for obstructions
    """

    for camera_index, camera_param in enumerate(camera_params):

        # loads the depth map
        depth_map = cv2.imread(
            os.path.join("data", config.experiment, camera_param['id'], f'{str(frame_id).zfill(10)}_rgbd.tiff'),
            cv2.IMREAD_ANYDEPTH)

        for person_data in data:
            # indicates whether one point of the face is visible
            one_point_is_visible = False
            # sums how many points are actually in the depth map, if none -> better set visible
            points_not_none = 0

            mesh_copy = copy.deepcopy(person_data["face_mesh"])
            # Points where we measure the distance. carefully selected 11-13 points of the mesh
            mesh_points_3d = np.asarray(mesh_copy.vertices)[indices.FACE_CHEEKS_UNIFORM_POINTS, :]

            mesh_points_2d = camera_utils.project_pose(mesh_points_3d, camera_param)

            # Masking only valid values
            mask_x = np.logical_and(mesh_points_2d[:, 0] >= 0, mesh_points_2d[:, 0] <= depth_map.shape[1])
            mask_y = np.logical_and(mesh_points_2d[:, 1] >= 0, mesh_points_2d[:, 1] <= depth_map.shape[0])
            mask = np.logical_and(mask_x, mask_y)

            mesh_points_2d = mesh_points_2d[mask, :]
            mesh_points_3d = mesh_points_3d[mask, :]
            points_3d = []
            for j, (mesh_point_2d, mesh_point_3d) in enumerate(zip(mesh_points_2d, mesh_points_3d)):
                depth_map_3d_point = camera_utils.reproject_pixel_in_3D(camera_param, mesh_point_2d, depth_map)
                points_3d.append(depth_map_3d_point)

                if isinstance(depth_map_3d_point, np.ndarray):
                    distance = np.linalg.norm(depth_map_3d_point - mesh_point_3d)
                    points_not_none += 1

                    if distance < 0.15:
                        one_point_is_visible = True
                        break

            person_data["is_visible"][camera_index] = one_point_is_visible

            # checks the angle between camera and the face
            camera_origin = (camera_param["color2world"] @ np.array([0, 0, 0, 1]))[:3]
            direction = (camera_param["color2world"] @ np.array([0, 0, 1, 1]))[:3]

            a = np.asarray(mesh_copy.vertices)[180]
            b = np.asarray(mesh_copy.vertices)[552]
            c = np.asarray(mesh_copy.vertices)[232]

            e = camera_origin[:2]
            f = direction[:2]
            a = a[:2]
            b = b[:2]
            c = c[:2]

            angle = get_angle_between_face_camera(a, b, c, e, f)

            # If the angle is between 180 - 20 and 180 + 20, the face is not visible
            if angle > (180 - 20) and angle < (180 + 20):
                person_data["is_visible"][camera_index] = False 


def get_angle_between_face_camera(a, b, c, e, f):
    """
    Calculates the angle between the viewing direction of the mesh and the camera

    :param a: (2) 2D coordinate of left side of the face mesh
    :param b: (2) 2D coordinate of right side of the mesh
    :param c: (2) 2D coordinate of nose of the mesh
    :param e: (2) position of camera
    :param f: (2) direction of the camera
    """
    # get direction c', point between a and b, which is perpendicular to c
    # https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    dx, dy = x2 - x1, y2 - y1
    det = dx * dx + dy * dy
    r = (dy * (y3 - y1) + dx * (x3 - x1)) / det
    c_prime = [x1 + r * dx, y1 + r * dy]

    def angle3pt(a, b, c):
        """Counterclockwise angle in degrees by turning from a to c around b
            Returns a float between 0.0 and 360.0"""
        ang = degrees(
            atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang


    L1 = [e, f]
    L2 = [c_prime, c]

    a = np.array(L2[0])
    b = np.array(L1[1])
    c = np.subtract(a, b)
    d = np.subtract(np.array(L2[1]), c)
    L2[1] = tuple(d)

    angle = angle3pt(L1[0], L1[1], L2[1])

    return angle
