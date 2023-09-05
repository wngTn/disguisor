import numpy as np
from probreg import filterreg
import open3d as o3d
import trimesh
import open3d.visualization.gui as gui

def register_head_mesh_to_pcd(head_mesh, face_mesh, pcd, max_iter_filterreg, sigma2, tol, w, voxel_size, max_iter_icp, threshold_icp):

    # Scaling factor for the head of the SMPL mesh for a better alignment
    scaling_factor = 1.17

    head_mesh_open3d = head_mesh.as_open3d
    head_mesh_open3d = head_mesh_open3d.scale(scaling_factor, center=head_mesh_open3d.get_center())

    face_mesh_open3d = face_mesh.as_open3d
    face_mesh_open3d = face_mesh_open3d.scale(scaling_factor, center=head_mesh_open3d.get_center())


    head_mesh_point_cloud_o3d = head_mesh_open3d.sample_points_uniformly(number_of_points=10000)
    head_mesh_point_cloud_o3d = head_mesh_open3d.sample_points_poisson_disk(
        number_of_points=1500, pcl=head_mesh_point_cloud_o3d)

    head_mesh_point_cloud = np.array(head_mesh_point_cloud_o3d.points, dtype=np.float32)

    source_pcd = voxelize_point_cloud(head_mesh_point_cloud, voxel_size)
    target_pcd, target_pcd_normals = voxelize_point_cloud(pcd, voxel_size, True)

    filterreg_tf_matrix = run_filterreg(source_pcd, target_pcd, target_pcd_normals, max_iter_filterreg, sigma2, tol, w)
    icp_tf_matrix = icp(source_pcd, target_pcd, max_iter_icp, threshold_icp, filterreg_tf_matrix)

    # app = gui.Application.instance
    # app.initialize()
    # vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    # vis.show_settings = True
    # head_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(head_mesh_point_cloud))
    # head_pcd.paint_uniform_color([1, 0, 0])
    # vis.add_geometry("head_point_cloud", head_pcd)
    # vis.add_geometry("pointcloud", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pcd)))
    # vis.add_geometry("aligned_head_point_cloud", head_pcd.transform(icp_tf_matrix))
    # app.add_window(vis)
    # app.run()

    face_mesh_open3d = face_mesh_open3d.transform(icp_tf_matrix)
    # The 226th vertex is the nose. That way we don't misalign the face
    # face_mesh_open3d = face_mesh_open3d.scale((1 / scaling_factor), np.asarray(face_mesh_open3d.vertices)[226])
    face_mesh_open3d = face_mesh_open3d.scale((1 / scaling_factor), center=head_mesh_open3d.get_center())

    head_mesh_open3d = head_mesh_open3d.transform(icp_tf_matrix)
    head_mesh_open3d = head_mesh_open3d.scale((1 / scaling_factor), center=head_mesh_open3d.get_center())

    head_mesh_trimesh = trimesh.Trimesh(vertices=np.array(head_mesh_open3d.vertices), faces=np.array(head_mesh_open3d.triangles), process=False)
    face_mesh_trimesh = trimesh.Trimesh(vertices=np.array(face_mesh_open3d.vertices), faces=np.array(face_mesh_open3d.triangles), process=False)

    return head_mesh_trimesh, face_mesh_trimesh, icp_tf_matrix


def icp(source_pcd, target_pcd, max_iter, threshold, trans_init):
    if (len(target_pcd) < 100):
        return trans_init

    threshold = 0.0375
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pcd)), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pcd)), threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

    return reg_p2p.transformation

def run_filterreg(source_pcd, target_pcd, target_pcd_normals, max_iter, sigma2, tol, w):
    tf_param = filterreg.registration_filterreg(source_pcd,
                                            target_pcd,
                                            objective_type='pt2pt',
                                            target_normals=np.asarray(
                                                target_pcd_normals),
                                            min_sigma2=0,
                                            maxiter=max_iter,
                                            sigma2=sigma2,
                                            w=w,
                                            tol=tol)

    tf_matrix = np.eye(4)
    tf_matrix[:3, :3] = tf_param[0].rot
    tf_matrix[:3, 3] = tf_param[0].t

    return tf_matrix

def voxelize_point_cloud(pcd, voxel_size, normals=False):
    """
    Voxelizes a point cloud and returns normals
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if normals:
        pcd.estimate_normals()
        return np.array(pcd.points, dtype=np.float32), np.array(pcd.normals, dtype=np.float32)
    return np.array(pcd.points, dtype=np.float32)


