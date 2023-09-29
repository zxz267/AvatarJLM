import numpy as np
import random
import glob
import pickle
from torch.utils.data import Dataset


class AMASS_Dataset(Dataset):
    """Motion Capture dataset"""
    def __init__(self, opt):
        self.opt = opt
        self.window_size = opt['window_size']
        self.num_input = opt['num_input']
        self.batch_size = opt['dataloader_batch_size']

        phase = self.opt['phase']
        dataroot = opt['dataroot']
        dataset_type = opt['dataset_type']
        assert dataset_type in ['amass_p1', 'amass_p2', 'amass_p3']

        if dataset_type == 'amass_p1':
            self.filename_list = glob.glob(f'./{dataroot}/*/{phase}/*.pkl')
        elif dataset_type == 'amass_p2':
            # TODO: all combinations.
            if phase == 'train':
                self.filename_list = glob.glob(f'./{dataroot}/MPI_HDM05/*/*.pkl') + glob.glob(f'./{dataroot}/BioMotionLab_NTroje/*/*.pkl')
            else:
                self.filename_list = glob.glob(f'./{dataroot}/CMU/*/*.pkl')
            # if phase == 'train':
            #     self.filename_list = glob.glob(f'./{dataroot}/CMU/*/*.pkl') + glob.glob(f'./{dataroot}/BioMotionLab_NTroje/*/*.pkl')
            # else:
            #     self.filename_list = glob.glob(f'./{dataroot}/MPI_HDM05/*/*.pkl')
            # if phase == 'train':
            #     self.filename_list = glob.glob(f'./{dataroot}/CMU/*/*.pkl') + glob.glob(f'./{dataroot}/MPI_HDM05/*/*.pkl')
            # else:
            #     self.filename_list = glob.glob(f'./{dataroot}/BioMotionLab_NTroje/*/*.pkl')
        elif dataset_type == 'amass_p3':
            self.filename_list = glob.glob(f'./{dataroot}/*/{phase}/*.pkl')

        print('-------------------------------number of {} data is {}'.format(phase, len(self.filename_list)))


    def __len__(self):
        return max(len(self.filename_list), self.batch_size)


    def __getitem__(self, idx):
        filename = self.filename_list[idx]
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        if self.opt['phase'] == 'train':
            while data['rotation_local_full_gt_list'].shape[0] < self.window_size:
                idx = random.randint(0, idx)
                filename = self.filename_list[idx]
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

        seq_len = data['hmd_position_global_full_gt_list'].shape[0]

        if self.opt['phase'] == 'train':
            start = np.random.randint(0, seq_len - self.window_size + 1)
            end = start + self.window_size
            return {'input_signal': data['hmd_position_global_full_gt_list'][start:end, ...].reshape(self.window_size, -1).float(),
                    'rotation_local_full': data['rotation_local_full_gt_list'][start:end,...].reshape(self.window_size, -1).float(), 
                    'body_param_list': 0,
                    'global_head_trans': 0, 
                    'pos_pelvis_gt': data['body_parms_list']['trans'][start:end],
                    'floor_height': data['offset_floor_height'],
                    'foot_contact': data['contacts'][:, [10, 11]][start:end] # left, right
                    }
        else:
            return {'input_signal': data['hmd_position_global_full_gt_list'].reshape(seq_len, -1).float(),
                    'rotation_local_full': data['rotation_local_full_gt_list'],
                    'body_param_list': data['body_parms_list'],
                    'global_head_trans':data['head_global_trans_list'],
                    'pos_pelvis_gt':data['body_parms_list']['trans'],
                    'floor_height': data['offset_floor_height'],
                    'foot_contact': data['contacts'][:, [10, 11]] # left, right
                    }


    