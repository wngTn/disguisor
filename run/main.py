from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm
import cv2
import numpy as np
np.random.seed(233)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
from utils.config import parse_config
from dataset.disguisor_dataset import DisguisORDataset
from core.registration import register_head_mesh_to_pcd
from core.render import render


def write_images(config, anonymized_data, frame_id):
    output_path = os.path.join("output", config.experiment)
    anonymized_images_path = os.path.join(output_path, "anonymized_images")
    os.makedirs(anonymized_images_path, exist_ok=True)
    for cn_index, image in enumerate(anonymized_data["anonymized_images"]):
        cv2.imwrite(os.path.join(anonymized_images_path, f"Frame_{frame_id}_cn0{cn_index + 1}.jpg"), image)

    if "anonymized_bbox_images" in anonymized_data:    
        anonymized_bbox_images_path = os.path.join(output_path, "anonymized_bbox_images")
        os.makedirs(anonymized_bbox_images_path, exist_ok=True)
        for cn_index, image in enumerate(anonymized_data["anonymized_bbox_images"]):
            cv2.imwrite(os.path.join(anonymized_bbox_images_path, f"Frame_{frame_id}_cn0{cn_index + 1}.jpg"), image)


def anonymize(config, data_loader):
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        batched_head_mesh_data, batched_face_mesh_data, batched_pcds_data, batched_textures, batched_frame_ids, batched_person_ids = data

        # Loop over each frame in the batch
        for frame_index, frame_person_ids in enumerate(batched_person_ids):
            # Loop over each person in this frame
            list_of_face_meshes_in_frame = []
            for person_id in frame_person_ids:
                head_mesh = batched_head_mesh_data[person_id][frame_index]
                face_mesh = batched_face_mesh_data[person_id][frame_index]
                pcd = batched_pcds_data[person_id][frame_index]
                texture = batched_textures[person_id][frame_index]
                frame_id = batched_frame_ids[frame_index]

                # Register the head mesh to the point cloud
                aligned_head_mesh, aligned_face_mesh, tf_matrix = register_head_mesh_to_pcd(head_mesh, face_mesh, pcd, config.max_iteration_filterreg, config.filterreg_sigma2, config.filterreg_tol, config.filterreg_w, config.voxel_size, config.max_iteration_icp, 0.0375)
                list_of_face_meshes_in_frame.append({
                    "face_mesh": aligned_face_mesh,
                    "texture": texture,
                    "person_id": person_id,
                })
            anonymized_data = render(config, list_of_face_meshes_in_frame, frame_id)
            write_images(config, anonymized_data, frame_id)


def main():
    config = parse_config()
    dataset = DisguisORDataset(config)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=DisguisORDataset.disguisOR_collate_fn)
    anonymize(config, data_loader)


if __name__ == "__main__":
    main()