from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from models.select_model import define_G
from models.model_base import ModelBase
from models.module import fk_module
from models.loss import velocityLoss, footContactLoss, penetrationLoss, footHeightLoss
from utils import utils_transform


class ModelAvatarJLM(ModelBase):
    def __init__(self, opt):
        super(ModelAvatarJLM, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        self.window_size = self.opt['netG']['window_size']
        self.bm = self.netG.module.body_model


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    def init_test(self):
        self.load(test=True)                           # load model
        self.log_dict = OrderedDict()         # log
    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self, test=False):
        load_path_G = self.opt['path']['pretrained_netG'] if test == False else self.opt['path']['pretrained']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['pred_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, test=False):
        # (batch, window_size, 54=18+18+9+9)
        self.input_signal = data['input_signal'].to(self.device)
        batch, seq_len = self.input_signal.shape[0], self.input_signal.shape[1]
        rotation = self.input_signal[:, :, :22*6].reshape(batch, seq_len, 22, 6)
        velocity_rotation = self.input_signal[:, :, 22*6:22*6*2].reshape(batch, seq_len, 22, 6)
        position = self.input_signal[:, :, 22*6*2:22*6*2+3*22].reshape(batch, seq_len, 22, 3)
        velocity_position = self.input_signal[:, :, 22*6*2+3*22:].reshape(batch, seq_len, 22, 3)
        self.input_signal = torch.cat((rotation, velocity_rotation, position, velocity_position), dim=3)


        # --------------------------------------
        self.gt_floor_height = data['floor_height'].to(self.device)
        self.gt_global_orientation = data['rotation_local_full'][:, :, :6].to(self.device)
        self.gt_joint_rotation = data['rotation_local_full'][:, :, 6:].to(self.device)
        self.gt_joint_position = fk_module(self.gt_global_orientation.reshape(batch * seq_len, -1), self.gt_joint_rotation.reshape(batch * seq_len, -1), self.bm).reshape(batch, seq_len, -1)

        # training only
        if not test:
            self.gt_floor_contact = data['foot_contact'].to(self.device)
            self.gt_global_root_trans = data['pos_pelvis_gt'].to(self.device)

        # testing only
        if test:
            self.gt_global_head_trans = data['global_head_trans'].to(self.device)
            self.gt_body_param_list = data['body_param_list']
            for k,v in self.gt_body_param_list.items():
                self.gt_body_param_list[k] = v.squeeze().to(self.device)
        
        
    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.predictions = self.netG(self.input_signal)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        loss = 0
        batch, seq_len, _ = self.predictions['pred_joint_position'][-1].shape

        for i in range(len(self.predictions['pred_global_orientation'])):
            global_orientation_loss = self.G_lossfn(self.predictions['pred_global_orientation'][i], self.gt_global_orientation)
            self.log_dict[f'global_orientation_loss_{i}'] = global_orientation_loss.item()
            loss += global_orientation_loss * 0.02

        for i in range(len(self.predictions['pred_joint_rotation'])):
            rotation_error_rate = 2
            joint_rotation_loss = self.G_lossfn(self.predictions['pred_joint_rotation'][i], self.gt_joint_rotation) * rotation_error_rate
            self.log_dict[f'joint_rotation_loss_{i}'] = joint_rotation_loss.item()
            loss += joint_rotation_loss

        for i in range(len(self.predictions['pred_joint_position'])):
            pose_error_rate = 5
            joint_position_loss = self.G_lossfn(self.predictions['pred_joint_position'][i][:, :, :22*3], self.gt_joint_position[:, :, :22*3]) * pose_error_rate
            loss += joint_position_loss
            self.log_dict[f'joint_position_loss_{i}'] = joint_position_loss.item()

        for i in range(len(self.predictions['pred_global_position'])):
            # global position
            gt_global_position = self.gt_joint_position[:, :, :22*3].reshape(batch, seq_len, 22, 3) + self.gt_global_root_trans[:, :, None]
            pred_global_position = self.predictions['pred_global_position'][i].reshape(batch, seq_len, 22, 3)

            velocity_error_scale = 50
            # velocity loss
            global_velocity_loss = velocityLoss(loss_func=self.G_lossfn, pred=pred_global_position, gt=gt_global_position) * velocity_error_scale
            loss += global_velocity_loss
            self.log_dict[f'global_velocity_loss_{i}'] = global_velocity_loss.item()

            global_velocity_loss = velocityLoss(loss_func=self.G_lossfn, pred=pred_global_position, gt=gt_global_position, interval=3) * velocity_error_scale / 3
            loss += global_velocity_loss
            self.log_dict[f'global_velocity_loss_i3_{i}'] = global_velocity_loss.item()

            global_velocity_loss = velocityLoss(loss_func=self.G_lossfn, pred=pred_global_position, gt=gt_global_position, interval=5) * velocity_error_scale / 5 
            loss += global_velocity_loss
            self.log_dict[f'global_velocity_loss_i5_{i}'] = global_velocity_loss.item()

            # foot contact loss
            foot_concat_error_scale = 20
            global_foot_concat_loss = footContactLoss(loss_func=self.G_lossfn, pred=pred_global_position, gt=gt_global_position) * foot_concat_error_scale
            loss += global_foot_concat_loss
            self.log_dict[f'global_foot_concat_loss_{i}'] = global_foot_concat_loss.item()

            # penetration loss
            pene_error_scale = 1
            pene_loss = penetrationLoss(self.predictions['pred_global_position'][i].reshape(batch * seq_len, -1, 3), self.gt_floor_height) * pene_error_scale
            loss += pene_loss
            self.log_dict[f'penetration_loss_{i}'] = pene_loss.item()

            # foot-height loss 
            foot_height_loss_scale = 0.5
            fh_loss = footHeightLoss(self.predictions['pred_global_position'][i].reshape(batch * seq_len, -1, 3), self.gt_floor_height, self.gt_floor_contact) * foot_height_loss_scale
            loss += fh_loss
            self.log_dict[f'foot_height_loss_{i}'] = fh_loss.item()

            # absolute only hand
            hand_alignment_loss_scale = 5
            hand_alignment_loss = self.G_lossfn(self.predictions['pred_global_position'][i].reshape(batch, seq_len, 22, 3)[:, :, [20, 21]], gt_global_position[:, :, [20, 21]]) * hand_alignment_loss_scale
            loss += hand_alignment_loss
            self.log_dict[f'hand_alignment_loss_{i}'] = hand_alignment_loss.item()


        for i, init_pose in enumerate(self.predictions['pred_init_pose']):
            init_pose_scale = 1
            gt_22 = self.gt_joint_position[:, :, :22*3].reshape(batch, seq_len, 22, 3)
            gt_22_head_centered = gt_22 - gt_22[:, :, 15:16]
            pose_loss = self.G_lossfn(init_pose, gt_22_head_centered.reshape(batch, seq_len, -1)) * init_pose_scale
            loss += pose_loss
            self.log_dict[f'joint_regress_loss_{i}'] = pose_loss.item()

        self.log_dict['total_loss'] = loss.item()
        loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        self.input_signal = self.input_signal.squeeze()
        self.gt_global_head_trans = self.gt_global_head_trans.squeeze()
        window_size = self.opt['datasets']['test']['window_size']
        with torch.no_grad():
            if self.input_signal.shape[0] <= window_size:
                pred_global_orientation_list = []  
                pred_joint_rotation_list = []   
                for frame_idx in range(0, self.input_signal.shape[0]):
                    outputs = self.netG(self.input_signal[0:frame_idx+1].unsqueeze(0))
                    pred_global_orientation, pred_joint_rotation = outputs['pred_global_orientation'][-1][:, -1], outputs['pred_joint_rotation'][-1][:, -1]
                    pred_global_orientation_list.append(pred_global_orientation)
                    pred_joint_rotation_list.append(pred_joint_rotation)
                pred_global_orientation_tensor = torch.cat(pred_global_orientation_list, dim=0)
                pred_joint_rotation_tensor = torch.cat(pred_joint_rotation_list, dim=0)
            else:  
                input_list_2 = []
                pred_global_orientation_list_1 = []  
                pred_joint_rotation_list_1 = []        
                # input with the frame number less than the window size
                for frame_idx in range(0, window_size):
                    outputs = self.netG(self.input_signal[0:frame_idx+1].unsqueeze(0))
                    pred_global_orientation, pred_joint_rotation = outputs['pred_global_orientation'][-1][:, -1], outputs['pred_joint_rotation'][-1][:, -1]
                    pred_global_orientation_list_1.append(pred_global_orientation)
                    pred_joint_rotation_list_1.append(pred_joint_rotation)
                pred_global_orientation_1 = torch.cat(pred_global_orientation_list_1, dim=0)
                pred_joint_rotation_1 = torch.cat(pred_joint_rotation_list_1, dim=0)
                    
                for frame_idx in range(window_size, self.input_signal.shape[0]):
                    input_list_2.append(self.input_signal[frame_idx-window_size+1:frame_idx+1,...].unsqueeze(0))
                input_tensor_2 = torch.cat(input_list_2, dim = 0)

                # divide into many parts to reduce memory
                part_size = 30
                part_num = (input_tensor_2.shape[0] - 1) // part_size + 1
                pred_global_orientation_list = [pred_global_orientation_1]
                pred_joint_rotation_list = [pred_joint_rotation_1]
                for part_idx in range(part_num):
                    outputs = self.netG(input_tensor_2[part_size * part_idx:min(part_size * (part_idx+1), input_tensor_2.shape[0])])
                    pred_global_orientation_this_part, pred_joint_rotation_this_part = outputs['pred_global_orientation'][-1][:, -1], outputs['pred_joint_rotation'][-1][:, -1]
                    pred_global_orientation_list.append(pred_global_orientation_this_part)
                    pred_joint_rotation_list.append(pred_joint_rotation_this_part)
                pred_global_orientation_tensor = torch.cat(pred_global_orientation_list, dim=0)
                pred_joint_rotation_tensor = torch.cat(pred_joint_rotation_list, dim=0)

        self.pred_global_orientation = pred_global_orientation_tensor
        self.pred_joint_rotation = pred_joint_rotation_tensor
        self.pred = torch.cat([pred_global_orientation_tensor, pred_joint_rotation_tensor],dim=-1).to(self.device)
        predicted_angle = utils_transform.sixd2aa(self.pred[:,:132].reshape(-1,6).detach()).reshape(self.pred[:,:132].shape[0],-1).float()

        # Calculate global translation
        T_head2world = self.gt_global_head_trans.clone()
        t_head2world = T_head2world[:,:3,3].clone()
        body_pose_local = self.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3]})
        position_global_full_local = body_pose_local.Jtr[:,:22,:]
        t_head2root = position_global_full_local[:,15,:]
        t_root2world = -t_head2root+t_head2world.cuda()

        self.predicted_body = self.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': t_root2world}) 
        self.predicted_vertex = self.predicted_body.v
        self.predicted_position = self.predicted_body.Jtr[:,:22,:]
        self.predicted_angle = predicted_angle
        self.predicted_translation = t_root2world

        self.gt_body = self.bm(**{k:v for k,v in self.gt_body_param_list.items() if k in ['pose_body','trans', 'root_orient']})
        self.gt_position = self.gt_body.Jtr[:,:22,:]
        self.gt_vertex = self.gt_body.v
        self.gt_local_angle = self.gt_body_param_list['pose_body']
        self.gt_global_translation = self.gt_body_param_list['trans']
        self.gt_global_orientation = self.gt_body_param_list['root_orient']


        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_prediction(self,):
        body_parms = OrderedDict()
        body_parms['pose_body'] = self.predicted_angle[...,3:66]
        body_parms['root_orient'] = self.predicted_angle[...,:3]
        body_parms['trans'] = self.predicted_translation
        body_parms['position'] = self.predicted_position
        body_parms['body'] = self.predicted_body
        body_parms['vertex'] = self.predicted_vertex
        return body_parms

    def current_gt(self, ):
        body_parms = OrderedDict()
        body_parms['pose_body'] = self.gt_local_angle
        body_parms['root_orient'] = self.gt_global_orientation
        body_parms['trans'] = self.gt_global_translation
        body_parms['position'] = self.gt_position
        body_parms['body'] = self.gt_body
        body_parms['floor_height'] = self.gt_floor_height
        body_parms['vertex'] = self.gt_vertex
        return body_parms


    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
