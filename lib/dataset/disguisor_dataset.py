from torch.utils.data import Dataset, DataLoader
import trimesh
import tqdm
import open3d as o3d
import numpy as np
import glob
import sys
import os
import json
import copy
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "lib"))

from utils.config import parse_config
import utils.indices as indices


def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    return data


def read_smpl(filename):
    datas = read_json(filename)
    outputs = []
    for data in datas:
        for key in ["Rh", "Th", "poses", "shapes", "expression"]:
            if key in data.keys():
                data[key] = np.array(data[key], dtype=np.float32)
        outputs.append(data)
    return outputs


class DisguisORDataset(Dataset):

    def __init__(self, config):
        self.data_dir = os.path.join("data", config.experiment)
        self.mesh_dir = os.path.join("input", config.experiment, "smpl_meshes")
        self.output_dir = os.path.join("output", config.experiment)

        self.texture_table = {}
        for i in range(len(config.texture_list)):
            self.texture_table[str(i)] = config.texture_list[i]

        self.texture_table[str(-1)] = config.default_texture

        self.camera_list = ["cn01", "cn02", "cn03", "cn04"]
        self.frame_ids = self._get_frame_ids()
        # self.frame_ids = [4010]

        self.len = len(self.frame_ids)

    def _get_frame_ids(self):
        all_files = glob.glob(self.data_dir + "/cn*/*.jpg", recursive=True)
        basenames = set(map(os.path.basename, all_files))
        frame_ids = sorted([int(x[:10]) for x in basenames])
        return frame_ids

    def _get_meshes(self, frame_id):
        mesh_frame_path = os.path.join(self.mesh_dir, f"{frame_id:010}")
        idxs = [x.split('.')[0] for x in os.listdir(mesh_frame_path)]

        meshes = {}
        for idx in idxs:
            mesh_path = os.path.join(mesh_frame_path, f"{idx}.obj")
            mesh_data = trimesh.load(mesh_path, process=False, maintain_order=True)
            meshes[idx] = mesh_data

        return meshes
    
    def _get_cropped_head_mesh(self, mesh_data):
        cropped_meshes = {}
        for idx, mesh in mesh_data.items():
            cropped_mesh = copy.deepcopy(mesh)
            # mask = np.zeros(6890, dtype=bool)
            # head_mask_keep = np.array(indices.HEAD)
            # mask[head_mask_keep] = True
            o3d_cropped_mesh = cropped_mesh.as_open3d
            o3d_cropped_mesh.remove_vertices_by_index(list(set(range(6890)) - set(indices.HEAD)))
            cropped_mesh = trimesh.Trimesh(vertices=np.array(o3d_cropped_mesh.vertices), faces=np.array(o3d_cropped_mesh.triangles), process=False)
            # cropped_mesh.update_vertices(mask)
            cropped_meshes[idx] = cropped_mesh

        return cropped_meshes

    def _get_cropped_face_mesh(self, mesh_data):
        cropped_meshes = {}
        for idx, mesh in mesh_data.items():
            cropped_mesh = copy.deepcopy(mesh)
            # Hack of converting to open3d then back to trimesh
            o3d_cropped_mesh = cropped_mesh.as_open3d
            o3d_cropped_mesh.remove_vertices_by_index(list(set(range(6890)) - set(indices.FACE_CHEEKS)))
            cropped_mesh = trimesh.Trimesh(vertices=np.array(o3d_cropped_mesh.vertices), faces=np.array(o3d_cropped_mesh.triangles), process=False)
            cropped_meshes[idx] = cropped_mesh

        return cropped_meshes

    def _get_cropped_point_cloud(self, frame_id, cropped_head_mesh_data):
        merged_pcd = o3d.geometry.PointCloud()

        for cam in self.camera_list:
            merged_pcd += o3d.io.read_point_cloud(
                os.path.join(self.data_dir, cam, f"{frame_id:04}_pointcloud.ply"))

        cropped_pcds = {}
        # crop the point cloud and voxel down according to the head mesh
        for idx, head_mesh in cropped_head_mesh_data.items():
            cropped_pcd = copy.deepcopy(merged_pcd)
            # crop the point cloud
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(head_mesh.vertices))
            bbox = bbox.scale(1.25, bbox.get_center())

            bbox.max_bound = bbox.max_bound + np.array([.1, .1, .1])
            bbox.min_bound = bbox.min_bound - np.array([.1, .1, .05])

            cropped_pcd = cropped_pcd.crop(bbox)
            cropped_pcds[idx] = np.array(cropped_pcd.points)

        return cropped_pcds

    def _get_textures(self, person_ids):
        textures = {}
        for person_id in person_ids:
            texture = cv2.imread(self.texture_table[person_id if person_id in self.texture_table else str(-1)])
            texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            textures[person_id] = texture

        return textures


    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]

        mesh_data =  self._get_meshes(frame_id)
        cropped_head_mesh_data = self._get_cropped_head_mesh(mesh_data)
        cropped_face_mesh_data = self._get_cropped_face_mesh(mesh_data)
        cropped_pcds_data = self._get_cropped_point_cloud(frame_id, cropped_head_mesh_data)
        person_ids = list(mesh_data.keys()) 
        textures = self._get_textures(person_ids)

        return cropped_head_mesh_data, cropped_face_mesh_data, cropped_pcds_data, textures, frame_id, person_ids


    def __len__(self):
        return self.len

    def disguisOR_collate_fn(batch):
        # Each item in the batch is a tuple: 
        # (cropped_head_mesh_data, cropped_face_mesh_data, cropped_pcds_data, textures, frame_id)

        # Initialize batched data structures
        batched_head_mesh_data = {}
        batched_face_mesh_data = {}
        batched_pcds_data = {}
        batched_textures = {}
        batched_frame_ids = []
        batched_person_ids = []

        # Loop through each item in the batch
        for item in batch:
            head_mesh_data, face_mesh_data, pcds_data, textures, frame_id, person_ids = item

            # Aggregate the meshes and point clouds
            for idx in head_mesh_data:
                if idx not in batched_head_mesh_data:
                    batched_head_mesh_data[idx] = []
                batched_head_mesh_data[idx].append(head_mesh_data[idx])

                if idx not in batched_face_mesh_data:
                    batched_face_mesh_data[idx] = []
                batched_face_mesh_data[idx].append(face_mesh_data[idx])

                if idx not in batched_pcds_data:
                    batched_pcds_data[idx] = []
                batched_pcds_data[idx].append(pcds_data[idx])

                if idx not in batched_textures:
                    batched_textures[idx] = []
                batched_textures[idx].append(textures[idx])

            # Append the textures and frame_ids to the batched lists
            batched_frame_ids.append(frame_id)
            batched_person_ids.append(person_ids)

        return batched_head_mesh_data, batched_face_mesh_data, batched_pcds_data, batched_textures, batched_frame_ids, batched_person_ids



if __name__ == "__main__":
    config = parse_config()
    dataset = DisguisORDataset(config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=DisguisORDataset.disguisOR_collate_fn)
    for batch in tqdm.tqdm(loader):
        a = batch
