import os
import argparse
import torch
from data.utils_data import process
from human_body_prior.body_model.body_model import BodyModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to data root.')
    parser.add_argument('--protocol', type=int, default=1, choices=[1, 2, 3], help='Prepare data mode.')
    parser.add_argument('--support_data', type=str, default='./support_data', help='Path to support data.')
    parser.add_argument('--data_split', type=str, default='./data/data_split', help='Path to data split.')
    cfg = parser.parse_args()

    bm_fname_male = os.path.join(cfg.support_data, 'body_models/smplh/{}/model.npz'.format('male'))
    dmpl_fname_male = os.path.join(cfg.support_data, 'body_models/dmpls/{}/model.npz'.format('male'))
    bm_fname_female = os.path.join(cfg.support_data, 'body_models/smplh/{}/model.npz'.format('female'))
    dmpl_fname_female = os.path.join(cfg.support_data, 'body_models/dmpls/{}/model.npz'.format('female'))

    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bm_male = BodyModel(bm_fname=bm_fname_male, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_male).to(device)
    bm_female = BodyModel(bm_fname=bm_fname_female, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_female).to(device)
    body_models = {'male': bm_male, 'female': bm_female} 

    if cfg.protocol in [1, 2]:
        dataset = ['MPI_HDM05', 'BioMotionLab_NTroje', 'CMU']
        for subset in dataset: 
            for phase in ['train', 'test']:
                print(subset, phase)
                split_file = os.path.join(cfg.data_split, subset, phase + "_split.txt")
                src = os.path.join(cfg.root, subset)
                dst = os.path.join("./data/protocol_1", subset, phase)
                os.makedirs(dst, exist_ok=True)
                process(src, dst, body_models, split_file)

    elif cfg.protocol in [3]:
        train_set = ['MPI_HDM05', 'BioMotionLab_NTroje', 'CMU', 'ACCAD', 'BMLmovi', 'EKUT', 'Eyes_Japan_Dataset', 'KIT', 'MPI_Limits', 'MPI_mosh', 'SFU', 'TotalCapture']
        test_set = ['HumanEva', 'Transitions_mocap']
        all_data = {**{k: 'train' for k in train_set}, **{k: 'test' for k in test_set}}
        for subset, phase in all_data.items(): 
            print(subset, phase)
            src = os.path.join(cfg.root, subset)
            dst = os.path.join("./data/protocol_3", subset, phase)
            os.makedirs(dst, exist_ok=True)
            process(src, dst, body_models)
