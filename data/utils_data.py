import numpy as np 
import torch
import os
import glob
import pickle
from utils import utils_transform
from sklearn.cluster import DBSCAN
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose

SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11, 
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]

DISCARD_TERRAIN_SEQUENCES = True # throw away sequences where the person steps onto objects (determined by a heuristic)
DISCARD_SHORTER_THAN = 1.0 # seconds

# for determining floor height
FLOOR_VEL_THRESH = 0.005
FLOOR_HEIGHT_OFFSET = 0.01
# for determining contacts
CONTACT_VEL_THRESH = 0.005 #0.015
CONTACT_TOE_HEIGHT_THRESH = 0.04
CONTACT_ANKLE_HEIGHT_THRESH = 0.08
# for determining terrain interaction
TERRAIN_HEIGHT_THRESH = 0.04 # if static toe is above this height
ROOT_HEIGHT_THRESH = 0.04 # if maximum "static" root height is more than this + root_floor_height
CLUSTER_SIZE_THRESH = 0.25 # if cluster has more than this faction of fps (30 for 120 fps)

def detect_joint_contact(body_joint_seq, joint_name, floor_height, vel_thresh, height_thresh):
    # calc velocity
    joint_seq = body_joint_seq[:, SMPL_JOINTS[joint_name], :]
    joint_vel = np.linalg.norm(joint_seq[1:] - joint_seq[:-1], axis=1)
    joint_vel = np.append(joint_vel, joint_vel[-1])
    # determine contact by velocity
    joint_contact = joint_vel < vel_thresh
    # compute heights
    joint_heights = joint_seq[:, 2] - floor_height
    # compute contact by vel + height
    joint_contact = np.logical_and(joint_contact, joint_heights < height_thresh)

    return joint_contact

def determine_floor_height_and_contacts(body_joint_seq, fps):
    '''
    Input: body_joint_seq N x 21 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    '''
    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, SMPL_JOINTS['hips'], :]
    left_toe_seq = body_joint_seq[:, SMPL_JOINTS['leftToeBase'], :]
    right_toe_seq = body_joint_seq[:, SMPL_JOINTS['rightToeBase'], :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]


    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)


    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
        # print(all_labels)
        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = min_median 
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

        if DISCARD_TERRAIN_SEQUENCES:
            # print(min_median + TERRAIN_HEIGHT_THRESH)
            # print(min_root_median + ROOT_HEIGHT_THRESH)
            for cluster_root_height, cluster_height, cluster_size in zip (cluster_root_heights, cluster_heights, cluster_sizes):
                root_above_thresh = cluster_root_height > (min_root_median + ROOT_HEIGHT_THRESH)
                toe_above_thresh = cluster_height > (min_median + TERRAIN_HEIGHT_THRESH)
                cluster_size_above_thresh = cluster_size > int(CLUSTER_SIZE_THRESH*fps)
                if root_above_thresh and toe_above_thresh and cluster_size_above_thresh:
                    discard_seq = True
                    print('DISCARDING sequence based on terrain interaction!')
                    break
    else:
        floor_height = offset_floor_height = 0.0

    # now find contacts (feet are below certain velocity and within certain range of floor)
    # compute heel velocities
    left_heel_seq = body_joint_seq[:, SMPL_JOINTS['leftFoot'], :]
    right_heel_seq = body_joint_seq[:, SMPL_JOINTS['rightFoot'], :]
    left_heel_vel = np.linalg.norm(left_heel_seq[1:] - left_heel_seq[:-1], axis=1)
    left_heel_vel = np.append(left_heel_vel, left_heel_vel[-1])
    right_heel_vel = np.linalg.norm(right_heel_seq[1:] - right_heel_seq[:-1], axis=1)
    right_heel_vel = np.append(right_heel_vel, right_heel_vel[-1])

    left_heel_contact = left_heel_vel < CONTACT_VEL_THRESH
    right_heel_contact = right_heel_vel < CONTACT_VEL_THRESH
    left_toe_contact = left_toe_vel < CONTACT_VEL_THRESH
    right_toe_contact = right_toe_vel < CONTACT_VEL_THRESH

    # compute heel heights
    left_heel_heights = left_heel_seq[:, 2] - floor_height
    right_heel_heights = right_heel_seq[:, 2] - floor_height
    left_toe_heights =  left_toe_heights - floor_height
    right_toe_heights =  right_toe_heights - floor_height

    left_heel_contact = np.logical_and(left_heel_contact, left_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    right_heel_contact = np.logical_and(right_heel_contact, right_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    left_toe_contact = np.logical_and(left_toe_contact, left_toe_heights < CONTACT_TOE_HEIGHT_THRESH)
    right_toe_contact = np.logical_and(right_toe_contact, right_toe_heights < CONTACT_TOE_HEIGHT_THRESH)

    contacts = np.zeros((num_frames, len(SMPL_JOINTS)))
    contacts[:,SMPL_JOINTS['leftFoot']] = left_heel_contact
    contacts[:,SMPL_JOINTS['leftToeBase']] = left_toe_contact
    contacts[:,SMPL_JOINTS['rightFoot']] = right_heel_contact
    contacts[:,SMPL_JOINTS['rightToeBase']] = right_toe_contact

    # hand contacts
    left_hand_contact = detect_joint_contact(body_joint_seq, 'leftHand', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_hand_contact = detect_joint_contact(body_joint_seq, 'rightHand', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:,SMPL_JOINTS['leftHand']] = left_hand_contact
    contacts[:,SMPL_JOINTS['rightHand']] = right_hand_contact

    # knee contacts
    left_knee_contact = detect_joint_contact(body_joint_seq, 'leftLeg', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_knee_contact = detect_joint_contact(body_joint_seq, 'rightLeg', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:,SMPL_JOINTS['leftLeg']] = left_knee_contact
    contacts[:,SMPL_JOINTS['rightLeg']] = right_knee_contact

    return offset_floor_height, contacts, discard_seq


def syn_acc(v, smooth_n=4):
    """
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0 and v.shape[0] >= 8:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def process(src, dst, body_models, split_file=None):
    assert src and dst
    
    rotation_local_full_gt_list = []
    hmd_position_global_full_gt_list = []
    body_parms_list = []
    head_global_trans_list = []

    if split_file is None:
        all_file = glob.glob(os.path.join(src ,'**', '*.npz'), recursive=True)
    else:
        with open(split_file, 'r') as f:
            all_file = ['/'.join(src.split('/')[:-1] + [line.rstrip('\n')]) for line in f]

    idx = 0
    for filepath in all_file:
        data = dict()
        bdata = np.load(filepath, allow_pickle=True)
        try:
            framerate = bdata["mocap_framerate"]
        except:
            print(filepath, list(bdata.keys()))
            continue 
        idx += 1

        if os.path.exists(os.path.join(dst, '{}.pkl'.format(idx))):
            continue
        
        stride = round(framerate / 60)

        bdata_poses = bdata["poses"][::stride,...]
        bdata_trans = bdata["trans"][::stride,...]
        subject_gender = bdata["gender"]
        body_model = body_models["male"]  # body_models[str(subject_gender)]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        body_parms = {
            'root_orient': torch.Tensor(bdata_poses[:, :3]).to(device),  # controls the global root orientation
            'pose_body': torch.Tensor(bdata_poses[:, 3:66]).to(device),  # controls the body
            'trans': torch.Tensor(bdata_trans).to(device),               # controls the global body position
        }

        body_parms_list = body_parms
        body_pose_world = body_model(**{k:v for k, v in body_parms.items() if k in ['pose_body', 'root_orient', 'trans']})
        body_pose_world.v = body_pose_world.v.cpu()
        body_pose_world.Jtr = body_pose_world.Jtr.cpu()

        output_aa = torch.Tensor(bdata_poses[:, :66]).reshape(-1,3)
        output_6d = utils_transform.aa2sixd(output_aa).reshape(bdata_poses.shape[0],-1)
        rotation_local_full_gt_list = output_6d[1:]
        rotation_local_matrot = aa2matrot(torch.tensor(bdata_poses).reshape(-1,3)).reshape(bdata_poses.shape[0],-1,9)
        rotation_global_matrot = local2global_pose(rotation_local_matrot, body_model.kintree_table[0].long()) # rotation of joints relative to the origin

        # pass very short sequence
        if body_pose_world.v.shape[0] <= 10:
            continue

        # -------------------------------- get synthetic IMU data ----------------------------------
        ji_mask = [18, 19, 4, 5, 15, 0]
        vi_mask = [1961, 5424, 1176, 4662, 411, 3021]
        out_grot = rotation_global_matrot[:, ji_mask]
        out_gacc = syn_acc(body_pose_world.v[:, vi_mask])
        # ------------------------------------------------------------------------------------------

        head_rotation_global_matrot = rotation_global_matrot[:,[15],:,:]
        rotation_global_6d = utils_transform.matrot2sixd(rotation_global_matrot.reshape(-1,3,3)).reshape(rotation_global_matrot.shape[0],-1,6)
        input_rotation_global_6d = rotation_global_6d[1:,:22,:]
        rotation_velocity_global_matrot = torch.matmul(torch.inverse(rotation_global_matrot[:-1]),rotation_global_matrot[1:])
        rotation_velocity_global_6d = utils_transform.matrot2sixd(rotation_velocity_global_matrot.reshape(-1,3,3)).reshape(rotation_velocity_global_matrot.shape[0],-1,6)
        input_rotation_velocity_global_6d = rotation_velocity_global_6d[:,:22,:]
        position_global_full_gt_world = body_pose_world.Jtr[:,:22,:] # position of joints relative to the world origin

        offset_floor_height, contacts, discard_seq = determine_floor_height_and_contacts(position_global_full_gt_world, 60)

        position_head_world = position_global_full_gt_world[:,15,:] # world position of head
        head_global_trans = torch.eye(4).repeat(position_head_world.shape[0],1,1)
        head_global_trans[:,:3,:3] = head_rotation_global_matrot.squeeze()
        head_global_trans[:,:3,3] = position_global_full_gt_world[:,15,:]

        head_global_trans_list = head_global_trans[1:]

        num_frames = position_global_full_gt_world.shape[0] - 1

        hmd_position_global_full_gt_list = torch.cat([
                                                                input_rotation_global_6d.reshape(num_frames,-1),
                                                                input_rotation_velocity_global_6d.reshape(num_frames,-1),
                                                                position_global_full_gt_world[1:, :22, :].reshape(num_frames,-1), 
                                                                position_global_full_gt_world[1:, :22, :].reshape(num_frames,-1)-position_global_full_gt_world[:-1, :22, :].reshape(num_frames,-1)], dim=-1)


        print(str(idx), framerate, src, len(all_file), bdata["poses"].shape[0], hmd_position_global_full_gt_list.shape)

        body_parms_list = {k: v[1:].cpu() for k, v in body_parms_list.items()}

        data['rotation_local_full_gt_list'] = rotation_local_full_gt_list.cpu()
        data['hmd_position_global_full_gt_list'] = hmd_position_global_full_gt_list.cpu()
        data['body_parms_list'] = body_parms_list
        data['head_global_trans_list'] = head_global_trans_list.cpu()
        data['framerate'] = 60
        data['gender'] = subject_gender
        data['filepath'] = filepath
        data['IMU_global_rotation'] = out_grot.cpu()[1:]
        data['IMU_global_acceleration'] = out_gacc.cpu()[1:]
        data['shape'] = bdata["betas"]
        data['offset_floor_height'] = offset_floor_height
        data['contacts'] = contacts[1:]

        with open(os.path.join(dst, '{}.pkl'.format(idx)), 'wb') as f:
            pickle.dump(data, f)

