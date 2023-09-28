import torch

def penetration_error(pred_mesh, floor_height):
    '''
    pred_mesh: (batch, v_num, 3)
    '''
    lowest_z, lowest_z_index = pred_mesh.min(1)
    lowest_z = lowest_z[:, 2]
    lowest_z_index = lowest_z_index[:, 2]
    floor_height = floor_height.float()
    lowest_z_filtered = torch.where(lowest_z>=floor_height, torch.FloatTensor([0]).to(pred_mesh.device), lowest_z)
    floor_height_filtered = torch.where(lowest_z>=floor_height, torch.FloatTensor([0]).to(pred_mesh.device), floor_height)
    seq_len = pred_mesh.shape[0] // floor_height.shape[0]
    floor_height = floor_height[:, None].repeat(1, seq_len).view(-1)
    return torch.abs(floor_height_filtered - lowest_z_filtered).mean()

def floating_error(pred_mesh, floor_height):
    '''
    pred_mesh: (batch, v_num, 3)
    '''
    lowest_z, lowest_z_index = pred_mesh.min(1)
    lowest_z = lowest_z[:, 2]
    lowest_z_index = lowest_z_index[:, 2]

    floor_height = floor_height.float()
    lowest_z_filtered = torch.where(lowest_z<=floor_height, torch.FloatTensor([0]).to(pred_mesh.device), lowest_z)
    floor_height_filtered = torch.where(lowest_z<=floor_height, torch.FloatTensor([0]).to(pred_mesh.device), floor_height)
    seq_len = pred_mesh.shape[0] // floor_height.shape[0]
    floor_height = floor_height[:, None].repeat(1, seq_len).view(-1)
    return torch.abs(floor_height_filtered - lowest_z_filtered).mean()

def skating_error(pred, gt):
    '''
    pred_mesh: (batch, v_num, 3)
    '''
    seq_len = pred.shape[0]
    batch = pred.shape[0] // seq_len

    # batch, seq_len = pred.shape[0], pred.shape[1]
    pred = pred.reshape(batch, seq_len, -1)
    gt = gt.reshape(batch, seq_len, -1)
    pred = pred[:, :, :22*3].reshape(batch, seq_len, 22, 3)
    gt = gt[:, :, :22*3].reshape(batch, seq_len, 22, 3)

    # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11
    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
    relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
    gt_joint_xyz = gt[:, :, relevant_joints, :]  # [BatchSize, 4, 3, Frames]
    gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, 1:, :, :] - gt_joint_xyz[:, :-1, :, :], dim=-1)  # [BatchSize, 4, Frames]
    fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=-1).repeat(1, 1, 1, 3)
    pred_joint_xyz = pred[:, :, relevant_joints, :]  # [BatchSize, 4, 3, Frames]
    pred_vel = pred_joint_xyz[:, 1:, :, :] - pred_joint_xyz[:, :-1, :, :]
    pred_vel[~fc_mask] = 0
    foot_concat_loss = torch.abs(torch.zeros(pred_vel.shape, device=pred_vel.device) - pred_vel).mean()
    return foot_concat_loss
