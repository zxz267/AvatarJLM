import random 
import collections
import torch
import torch.nn as nn
from utils.utils_transform import sixd2matrot, matrot2sixd
from models.module import fk_module, trunc_normal_, local2global_pose


class SimpleSMPL(nn.Module):
    def __init__(self, body_model, joint_regressor_dim=1024, embed_dim=1024):
        super(SimpleSMPL, self).__init__()
        self.regressor = nn.Sequential(
                            nn.Linear(joint_regressor_dim, embed_dim),
                            nn.LeakyReLU(0.1),
                            nn.Linear(embed_dim, 22*6))
        self.body_model = body_model
    
    def forward(self, x):
        params = self.regressor(x)
        global_orientation = params[:, :, :6]
        joint_rotation = params[:, :, 6:]
        joint_position = fk_module(global_orientation.reshape(-1, 6), joint_rotation.reshape(-1, 21*6), self.body_model)
        joint_position = joint_position[:, :22]
        return joint_position, params

class AlternativeST(nn.Module):
    def __init__(self, repeat_time=1, s_layer=2, t_layer=2, embed_dim=256, nhead=8):
        super(AlternativeST, self).__init__()
        self.num_layer = repeat_time
        self.s_layer = s_layer
        self.t_layer = t_layer
        self.STB = nn.ModuleList()
        self.TTB = nn.ModuleList()
        for _ in range(repeat_time):
            if self.s_layer != 0:
                spatial_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True)
                self.STB.append(nn.TransformerEncoder(spatial_layer, num_layers=s_layer))
            if self.t_layer != 0:
                temporal_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True)
                self.TTB.append(nn.TransformerEncoder(temporal_layer, num_layers=t_layer))

    def forward(self, feat):
        assert len(feat.shape) == 4, 'The input shape dimension should be 4, e.g., (batch, seq_len, joint_num, feat_dim).'
        batch, seq_len, joint_num, feat_dim = feat.shape
        input_token = feat[:, :, -1].clone().detach()
        for i in range(self.num_layer):
            if joint_num == 45:  # use EIF-token
                if self.s_layer != 0:
                    feat = self.STB[i](feat.reshape(batch*seq_len, joint_num, -1)).reshape(batch, seq_len, joint_num, -1)
                    feat[:, :, -1] = input_token
                if self.t_layer != 0:
                    feat = self.TTB[i](feat.reshape(batch, seq_len, joint_num, -1).permute(0, 2, 1, 3).reshape(batch*joint_num, seq_len, -1)).reshape(batch, joint_num, seq_len, -1).permute(0, 2, 1, 3)
                    feat[:, :, -1] = input_token
            else:
                feat = self.STB[i](feat.reshape(batch*seq_len, joint_num, -1)).reshape(batch, seq_len, joint_num, -1)
                feat = self.TTB[i](feat.reshape(batch, seq_len, joint_num, -1).permute(0, 2, 1, 3).reshape(batch*joint_num, seq_len, -1)).reshape(batch, joint_num, seq_len, -1).permute(0, 2, 1, 3)
        return feat

class MotionNet(nn.Module):
    def __init__(self, body_model, num_layer=6, s_layer=1, t_layer=1, joint_feat_dim=256, node_num=22, reg_hidden_dim=1024, spatial_embedding=True, temporal_embedding=True, nhead=8):
        super(MotionNet, self).__init__()
        self.body_model = body_model
        self.use_spatial_embedding = spatial_embedding
        self.use_temporal_embedding = temporal_embedding
        self.transformer = AlternativeST(repeat_time=num_layer, s_layer=s_layer, t_layer=t_layer, embed_dim=joint_feat_dim, nhead=nhead) 
        if self.use_temporal_embedding:
            max_seq_len = 200
            self.temp_embed = nn.Parameter(torch.zeros(1, max_seq_len, 1, joint_feat_dim))
            trunc_normal_(self.temp_embed, std=.02)
        if self.use_spatial_embedding:
            max_joint_num = node_num 
            self.joint_position_embed = nn.Parameter(torch.zeros(1, 1, max_joint_num, joint_feat_dim))
            trunc_normal_(self.joint_position_embed, std=.02)

        # ------------------regression head-------------------------
        self.param_regressor = nn.Sequential(
                            nn.Linear(joint_feat_dim * node_num, reg_hidden_dim),
                            nn.GroupNorm(8, reg_hidden_dim),
                            nn.LeakyReLU(0.1),
                            nn.Linear(reg_hidden_dim, 22*6)
            )


    def head_align(self, joint_position, gt_head_position):
        head2root = joint_position[:, :, 15].clone()
        root_transl = gt_head_position - head2root 
        global_position = root_transl[:, :, None] + joint_position
        return global_position, root_transl


    def forward(self, joint_embedding, gt_head_position):
        batch, seq_len = joint_embedding.shape[0], joint_embedding.shape[1]
        # add spatial positional embedding
        if self.use_spatial_embedding:
            joint_embedding = joint_embedding + self.joint_position_embed
        # add temporal positional embedding
        if self.use_temporal_embedding:
            joint_embedding = joint_embedding + self.temp_embed[:,:seq_len,:,:]
        ST_feat = self.transformer(joint_embedding)
        param = self.param_regressor(ST_feat.reshape(batch * seq_len, -1)).reshape(batch, seq_len, -1)
        global_orientation, joint_rotation = param[:, :, :6], param[:, :, 6:]
        joint_position = fk_module(global_orientation.reshape(-1, 6), joint_rotation.reshape(-1, 21*6), self.body_model)[:, :22].reshape(batch, seq_len, 22, 3)
        global_position, root_transl = self.head_align(joint_position, gt_head_position)
        return global_orientation, joint_rotation, joint_position.reshape(batch, seq_len, -1), global_position


class AvatarJLM(nn.Module):
    def __init__(
        self, 
        body_model,
        nhead=8,
        input_dim=22*18, 
        embed_dim=1024, 
        single_frame_feat_dim=1024, 
        joint_regressor_dim=1024, 
        joint_embed_dim=256,
        mask_training=False,
        replace=True,
        position_token=True,
        rotation_token=True,
        input_token=True
        ):
        super(AvatarJLM, self).__init__()
        # ----------------- general setting ---------------------
        self.body_model = body_model
        self.mask_training = mask_training
        if self.mask_training:
            print('[Model Info] Use mask training.')
        # ----------------- general setting ---------------------

        # ----------------- stage-1 ---------------------
        self.linear_embedding_static = nn.Linear(input_dim, embed_dim)
        self.joint_regressor = SimpleSMPL(self.body_model, joint_regressor_dim)
        self.gt_head_hand_replace = replace
        if self.gt_head_hand_replace:
            print('[Model Info] Use GT tracking signals replacement.')
        # ----------------- stage-1 ---------------------

        # ----------------- stage-2 ---------------------
        self.token_num = 0
        self.use_position_token = position_token
        self.use_rotation_token = rotation_token
        self.use_input_token = input_token
        if self.use_position_token:
            print('[Model Info] Use position token.')
            self.token_num += 22
            self.position_embedding = nn.Linear(3, joint_embed_dim * 2)
        if self.use_rotation_token:
            print('[Model Info] Use rotation token.')
            self.token_num += 22
            self.rotation_embedding = nn.Linear(6, joint_embed_dim * 2)
        if self.use_input_token:
            print('[Model Info] Use input token.')
            self.token_num += 1
            self.input_token_embedding = nn.Linear(single_frame_feat_dim, joint_embed_dim * 2)
        self.motion_net = MotionNet(body_model, joint_feat_dim=joint_embed_dim * 2, node_num=self.token_num, nhead=nhead)
        print(f'[Model Info] Total token number is {self.token_num}.')
        # ----------------- stage-2 ---------------------


    def input_processing(self, input_tensor):
        # --------- get input relative information --------- 
        processed_info = {}
        processed_info['global_gt_head_hand_rotation'] = input_tensor[:, :, [15, 20, 21], 0:6].clone()
        processed_info['global_gt_head_hand_position'] = input_tensor[:, :, [15, 20, 21], 12:15].clone()
        processed_info['global_gt_head_position'] = processed_info['global_gt_head_hand_position'][:, :, 0].clone()
        processed_info['relative_gt_head_hand_position'] = processed_info['global_gt_head_hand_position'] - processed_info['global_gt_head_position'][:, :, None]
        # --------- set masked input ------------
        selected_joint_index = list(range(0, 14+1)) + list(range(16, 19+1))
        input_tensor[:, :, selected_joint_index] = torch.ones_like(input_tensor[:, :, selected_joint_index]).cuda() * 0.01
        return input_tensor, processed_info


    def random_mask(self, joint_embedding):
        batch, seq_len = joint_embedding.shape[0], joint_embedding.shape[1]
        for b_idx in range(batch):
            for s_idx in range(seq_len):
                selected_joint_index = list(range(0, 14+1)) + list(range(16, 19+1))
                if self.use_rotation_token:
                    selected_joint_index += list(range(0+22, 14+1+22)) + list(range(16+22, 19+1+22))
                mask_num = 2
                random.shuffle(selected_joint_index)
                mask_index = selected_joint_index[:mask_num]
                joint_embedding[b_idx, s_idx, mask_index] = torch.ones_like(joint_embedding[b_idx, s_idx, mask_index]).cuda() * 0.01
        return joint_embedding


    def tokenize(self, relative_pose_init, rotation_init, static_embedding):
        batch, seq_len = static_embedding.shape[0], static_embedding.shape[1]
        joint_embedding_list = []
        if self.use_position_token:
            position_token = self.position_embedding(relative_pose_init.reshape(batch * seq_len, 22, -1))
            joint_embedding_list.append(position_token)
        if self.use_rotation_token:
            rotation_token = self.rotation_embedding(rotation_init.reshape(batch * seq_len, 22, -1))
            joint_embedding_list.append(rotation_token)
        if self.use_input_token:
            input_token = self.input_token_embedding(static_embedding.detach()).reshape(batch*seq_len, -1)[:, None]
            joint_embedding_list.append(input_token)
        assert len(joint_embedding_list) > 0
        joint_embedding = joint_embedding_list[-1] if len(joint_embedding_list) == 1 else torch.cat(joint_embedding_list, dim=-2)
        return joint_embedding


    def stage_1(self, input_tensor):
        features = {}
        batch, seq_len, _, _ = input_tensor.shape
        input_tensor, processed_info = self.input_processing(input_tensor)
        static_embedding = self.linear_embedding_static(input_tensor.reshape(batch, seq_len, -1))
        relative_pose_init, rotation_init = self.joint_regressor(static_embedding)
        rotation_init = rotation_init.reshape(batch, seq_len, 22, 6)
        relative_pose_init = relative_pose_init.reshape(batch, seq_len, 22, 3)
        # to head-relative
        head_index = 15
        relative_pose_init = relative_pose_init - relative_pose_init[:, :, head_index:head_index+1]
        rotation_init = rotation_init.clone()
        # rotation to global rotation
        rotation_local_matrot = sixd2matrot(rotation_init.reshape(-1, 6)).reshape(batch * seq_len, 22, 9)
        rotation_global_matrot = local2global_pose(rotation_local_matrot, self.body_model.kintree_table[0].long()) # rotation of joints relative to the origin
        rotation_init = matrot2sixd(rotation_global_matrot.reshape(-1, 3, 3)).reshape(batch, seq_len, 22, 6)
        # gt replace
        if self.gt_head_hand_replace:
            relative_pose_init[:, :, [15, 20, 21]] = processed_info['relative_gt_head_hand_position']
            rotation_init[:, :, [15, 20, 21]] = processed_info['global_gt_head_hand_rotation']
        relative_pose_init = relative_pose_init.reshape(batch, seq_len, -1)
        features['relative_pose_init'] = relative_pose_init
        features['rotation_init'] = rotation_init
        features['static_embedding'] = static_embedding
        return features, processed_info


    def stage_2(self, features, processed_info):
        batch, seq_len = features['static_embedding'].shape[0], features['static_embedding'].shape[1]
        joint_embedding = self.tokenize(features['relative_pose_init'], features['rotation_init'], features['static_embedding'])
        # --------------- mask training ---------------
        if self.training and self.mask_training:
            joint_embedding = self.random_mask(joint_embedding.reshape(batch, seq_len, self.token_num, -1))
        global_orientation, joint_rotation, joint_position, global_position = self.motion_net(joint_embedding.reshape(batch, seq_len, self.token_num, -1), processed_info['global_gt_head_position'])
        return global_orientation, joint_rotation, joint_position, global_position


    def forward(self, input_tensor):
        outputs = collections.defaultdict(list)
        features, processed_info = self.stage_1(input_tensor)
        global_orientation, joint_rotation, joint_position, global_position = self.stage_2(features, processed_info)
        outputs['pred_init_pose'].append(features['relative_pose_init'])
        outputs['pred_global_orientation'].append(global_orientation)
        outputs['pred_joint_rotation'].append(joint_rotation)
        outputs['pred_joint_position'].append(joint_position)
        outputs['pred_global_position'].append(global_position)
        return outputs
