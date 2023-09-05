import pickle
import numpy as np
import os
import shutil
import json
from glob import glob
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as cl

'''
This file converts the output of voxelpose to the output of easymocap
It tracks the people through easy euclidian distance between each frame
'''


NAME = 'simulation'

def prepare_out_dirs(prefix, dataDir='keypoints3d'):
    output_dir = os.path.join(prefix, dataDir)
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def convToEasyMocap(preds):
    output_dir = prepare_out_dirs(prefix=os.path.join(f'output_easymocap_{NAME}', f"{NAME}"))
    for j, frame in enumerate(preds):
        frame_num = '{:06d}'.format(j)
        file_name = frame_num + '.json'
        file_path= os.path.join(output_dir, file_name)
        for pred in frame:
                pred['id'] = int(pred['id'])
                # pred['keypoints3d'][:, :3] = pred['keypoints3d'][:, :3] / 1000
                pred['keypoints3d'] = [[round(c, 3) for c in row] for row in pred['keypoints3d'].tolist()]

        json_string = json.dumps(frame, indent=4)
        with open(file_path, 'w') as outfile:
            outfile.write(json_string)
            print('Saved:', file_path)


# def distance_two_people_2(p1, p2):
#     """
#     Calculates the distance of two people based on where all their keypoints are

#     :param p1: _description_
#     :param p2: _description_
#     """


def distance_two_people(p1, p2):
    """
    Calculates the distance of two people based on where their hips are:
    (joints_3d_u[:, 11] + joints_3d_u[:, 12]) / 2.0

    :param p1: keypoints of person 1
    :param p2: keypoints of person 2
    """

    hip_p1 = (p1[11, :] + p1[12, :]) / 2.0
    hip_p2 = (p2[11, :] + p2[12, :]) / 2.0

    distance = np.linalg.norm(hip_p1 - hip_p2)

    return distance


def main():
    res = []
    with open(os.path.join('visualized_output', NAME, 'pred_voxelpose.pkl'), 'rb') as f:
        preds = pickle.load(f)
        first_frame = preds[5]

        # # list of the positions of all the people in the first frame
        # # people:
        # # {
        # # <id> : previous_keypoints
        # # <id> : previous_keypoints
        # # }
        # people = {}
        # for i in range(len(first_frame)):
        #     people[i] = first_frame[i][:, :3]

        people = {}

        # we start with second frame
        for i, (k, v) in enumerate(preds.items()):
            v = np.array(v)

            if len(v) == 0:
                res.append([])
                continue


            dist_matrix = []
            pred_kps = v[:, :, :3]

            for pred in pred_kps:
                # calculate distance with first person, second person and so on
                # dist_row = [dist_to_id_0, dist_to_id_1, dist_to_id_2]
                dist_row = []
                for persond_id, predecessor_kps in people.items():
                    dist_row.append(distance_two_people(pred, predecessor_kps))

                dist_matrix.append(dist_row)
            
            # dist_matrix = [
            #   [dist_first_person_to_id_0, dist_first_person_to_id_1, ...]
            #   [dist_second_person_to_id_0, dist_second_person_to_id_1,...]
            # ]
            dist_matrix = np.array(dist_matrix)

            # dict that saves the person_ids and respective ids of the <v> of the people in the current frame
            current_frame = {}
            for person_id in range(dist_matrix.shape[1]):
                # the nearest person of <person_id> is <nearest_person_id>
                nearest_person_id = np.argmin(dist_matrix[:, person_id])

                # check if the <person_id> already has a nearest person
                if nearest_person_id not in list(current_frame.values()):
                    # if not put <person_id> : kps_previous frame to the dict
                    current_frame[person_id] = nearest_person_id
                else:
                    # <person_id> already has a nearest person
                    # check if the new nearest_person_id is even nearer
                    if dist_matrix[nearest_person_id, person_id] < dist_matrix[nearest_person_id, 
                        list(current_frame.keys())[list(current_frame.values()).index(nearest_person_id)] # key of nearest_person_id
                    ]:
                        # if so, put it in the current frame
                        del current_frame[list(current_frame.keys())[list(current_frame.values()).index(nearest_person_id)]]
                        current_frame[person_id] = nearest_person_id
            
            # current_fame = {
            # <person_id> : kp_id,
            # <person_id> : kp_id,
            # }

            # check residual kp_ids with the person_id. If there is an entry in "people". If so, update
            for kp_id in range(len(pred_kps)):
                # check if it's actual residual
                if kp_id not in list(current_frame.values()):

                    dist_kp_id_other_people = {}

                    # check every person_id that is not in current_frame
                    for p_id, kps in people.items():
                        if p_id not in current_frame:

                            dist = distance_two_people(pred_kps[kp_id], kps)
                            dist_kp_id_other_people[p_id] = dist


                    if len(people) == 0:
                        current_frame[kp_id] = kp_id

                    else: 
                        try: 
                            min_key_dist_kp_id_other_people = min(dist_kp_id_other_people, key=dist_kp_id_other_people.get)

                            # There is already an entry, update it
                            if dist_kp_id_other_people[min_key_dist_kp_id_other_people] < 0.05:
                                current_frame[min_key_dist_kp_id_other_people] = kp_id

                        except:
                            current_frame[len(people)] = kp_id


            # update people and write result
            result_frame = []
            for kk, vv in current_frame.items():
                people[kk] = pred_kps[vv]

                # write to result
                res_temp = {
                    'id': kk,
                    'keypoints3d': v[vv]
                }
                result_frame.append(res_temp)
            
            res.append(result_frame)
        convToEasyMocap(res)


def no_tracking():
    res = []
    with open(os.path.join('visualized_output', f"{NAME}", f'pred_{NAME}.pkl'), 'rb') as f:
        preds = pickle.load(f)
        
        for i, (k, v) in enumerate(preds.items()):
            # update people and write result
            result_frame = []
            for j, vv in enumerate(v):
                # write to result
                res_temp = {
                    'id': j,
                    'keypoints3d': vv
                }
                result_frame.append(res_temp)
            
            res.append(result_frame)
        convToEasyMocap(res)


if __name__=='__main__':
    #main()
    no_tracking()

# _____________ VISUALIZATION __________________

def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    return data


def read_keypoints3d(filename):
    data = read_json(filename)
    res_ = []
    for d in data:
        pid = d['id'] if 'id' in d.keys() else d['personID']
        pose3d = np.array(d['keypoints3d'], dtype=np.float32)
        if pose3d.shape[0] > 25:
            # 对于有手的情况，把手的根节点赋值成body25上的点
            pose3d[25, :] = pose3d[7, :]
            pose3d[46, :] = pose3d[4, :]
        if pose3d.shape[1] == 3:
            pose3d = np.hstack([pose3d, np.ones((pose3d.shape[0], 1))])
        res_.append({
            'id': pid,
            'keypoints3d': pose3d
        })
    return res_


# def read():
#     k3dpath = 'output_easymocap_trial_08_recording_04/trial_08_recording_04/keypoints3d'
#     filenames = sorted(glob(k3dpath + '/*.json'))
#     results = []
#     for nf, filename in enumerate(filenames):
#         basename = os.path.basename(filename)
#         infos = read_keypoints3d(filename)

#         results.append(infos)
#     return results

# LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
#         [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]

# def _visualize(preds, ids, file_name):
#     file_name = 'tmp/' + file_name + "_3d.png"

#     # preds = preds.cpu().numpy()
#     batch_size = 1
#     xplot = min(4, batch_size)
#     yplot = int(math.ceil(float(batch_size) / xplot))

#     width = 4.0 * xplot
#     height = 4.0 * yplot
#     fig = plt.figure(0, figsize=(width, height))
#     plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
#                         top=0.95, wspace=0.05, hspace=0.15)

#     colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
#     ax = plt.subplot(yplot, xplot, 1, projection='3d')
#     for pred_id, pred in zip(ids, preds):
#         joint = pred # joint of one person
#         for k in eval("LIMBS{}".format(len(pred))):
#             x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
#             y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
#             z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
#             ax.plot(x, y, z, c=colors[pred_id], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
#                     markeredgewidth=1)
#     print('Wrote', file_name)
#     plt.savefig(file_name)
#     plt.close(0)

# def visualize():
#     results = read()
#     for i in range(len(results)):
#         preds = np.stack([info['keypoints3d'] for info in results[i]])
#         ids = np.stack([info['id'] for info in results[i]])
#         file_name = str(i).zfill(4)
#         _visualize(preds, ids, file_name)



    #visualize()