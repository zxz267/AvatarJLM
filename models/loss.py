import torch
           

def velocityLoss(loss_func, pred, gt, interval=1):
    '''
    pred: shape=(batch, seq_len, joint_num, 3)
    '''
    batch, seq_len = pred.shape[0], pred.shape[1]
    pred = pred.reshape(batch, seq_len, -1)
    gt = gt.reshape(batch, seq_len, -1)
    target_vel = gt[:, interval::interval, :22*3] - gt[:, :-interval:interval, :22*3]
    pred_vel = pred[:, interval::interval, :22*3] - pred[:, :-interval:interval, :22*3] 
    velocity_loss = loss_func(target_vel, pred_vel)
    return velocity_loss
    
           
def footContactLoss(loss_func, pred, gt):
    '''
    '''
    batch, seq_len = pred.shape[0], pred.shape[1]
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
    foot_concat_loss = loss_func(torch.zeros(pred_vel.shape, device=pred_vel.device), pred_vel)
    return foot_concat_loss


def penetrationLoss(pred_mesh, floor_height):
    '''
    pred_mesh: (batch, v_num, 3)
    '''
    seq_len = pred_mesh.shape[0] // floor_height.shape[0]
    floor_height = floor_height[:, None].repeat(1, seq_len).view(-1).float()
    lowest_z, lowest_z_index = pred_mesh.min(1)
    lowest_z = lowest_z[:, 2]
    lowest_z_index = lowest_z_index[:, 2]
    lowest_z_filtered = torch.where(lowest_z>=floor_height, torch.FloatTensor([0]).to(pred_mesh.device), lowest_z)
    floor_height_filtered = torch.where(lowest_z>=floor_height, torch.FloatTensor([0]).to(pred_mesh.device), floor_height)
    return torch.abs(floor_height_filtered - lowest_z_filtered).mean()


def footHeightLoss(pred_mesh, floor_height, foot_contact):
    '''
    '''
    seq_len = pred_mesh.shape[0] // floor_height.shape[0]
    # (batch*seq, 2)
    floor_height = floor_height[:, None].repeat(1, seq_len).view(-1).float()[:, None].repeat(1, 2)
    # (batch*seq, 2)
    foot_height = pred_mesh[:, [10, 11], 2]
    # (batch*seq, 2)
    foot_contact = foot_contact.reshape(-1, 2)
    return (torch.abs(foot_height - floor_height) * foot_contact).mean()

