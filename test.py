import os 
import os.path
import argparse
import logging
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils.utils_metric import penetration_error, floating_error, skating_error
from utils import utils_visualize as vis


def evaluate(opt, logger, model, test_loader, epoch=0, save_animation=False):
    error_dict = defaultdict(list)
    for index, test_data in enumerate(test_loader):
        logger.info("testing the sample {}/{}".format(index, len(test_loader)))
        model.feed_data(test_data, test=True)
        model.test()

        body_parms_pred = model.current_prediction()
        body_parms_gt = model.current_gt()
        predicted_angle = body_parms_pred['pose_body']
        predicted_position = body_parms_pred['position']
        gt_angle = body_parms_gt['pose_body']
        gt_position = body_parms_gt['position']
        predicted_angle = predicted_angle.reshape(body_parms_pred['pose_body'].shape[0],-1,3)                    
        gt_angle = gt_angle.reshape(body_parms_gt['pose_body'].shape[0],-1,3)

        # vis
        if save_animation:
            if index in [0, 10, 20]:
                print(f'[INFO] Saving animation of sequence-{index}...')
                video_dir = os.path.join(opt['path']['images'], str(index))
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)
                save_video_path_gt = os.path.join(video_dir, 'gt.avi')
                if not os.path.exists(save_video_path_gt):
                    vis.save_animation(body_pose=body_parms_gt['body'], savepath=save_video_path_gt, bm = model.bm, fps=60, resolution = (800,800))
                save_video_path = os.path.join(video_dir, '{:d}.avi'.format(epoch))
                vis.save_animation(body_pose=body_parms_pred['body'], savepath=save_video_path, bm = model.bm, fps=60, resolution = (800,800))

        # local position error
        pos_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))) * 100
        error_dict['MPJPE'].append(pos_error_)

        # rotation error
        rot_error_ = torch.mean(torch.absolute(gt_angle-predicted_angle)) * 57.2958
        error_dict['MPJRE'].append(rot_error_)

        # velocity error
        gt_velocity = (gt_position[1:,...] - gt_position[:-1,...]) * 60
        predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...]) * 60
        vel_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1))) * 100
        error_dict['MPJVE'].append(vel_error_)

        # jitter - prediction
        joint_p = predicted_position.reshape(-1, 22, 3)
        f = 60
        jitter_pred_ = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2).mean() / 100
        error_dict['Jitter'].append(jitter_pred_)

        # jitter - gt
        joint_t = gt_position.reshape(-1, 22, 3)
        jitter_gt_ = ((joint_t[3:] - 3 * joint_t[2:-1] + 3 * joint_t[1:-2] - joint_t[:-3]) * (f ** 3)).norm(dim=2).mean() / 100
        error_dict['Jitter_GT'].append(jitter_gt_)

        # penetration error 
        pen_error_ = penetration_error(predicted_position, body_parms_gt['floor_height']) * 100
        error_dict['Penetration'].append(pen_error_)

        # floating error 
        float_error_ = floating_error(predicted_position, body_parms_gt['floor_height']) * 100
        error_dict['Floating'].append(float_error_)
        
        # ground error 
        error_dict['Ground'].append(pen_error_ + float_error_)

        # skating error
        skate_error_ = skating_error(predicted_position, gt_position) * 100
        error_dict['Skate'].append(skate_error_)

        # local hands position error
        pos_error_hands_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[...,[20,21]]) * 100
        error_dict['H-PE'].append(pos_error_hands_)

        # upper joints error
        upper_joint_index = [0, 3, 6, 9, 13, 14, 12, 15, 16, 17, 18, 19, 20, 21]
        upper_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[:, upper_joint_index]) * 100
        error_dict['U-PE'].append(upper_error_)

        # lower joints error
        lower_joint_index = [1, 2, 4, 5, 7, 8, 10, 11]
        lower_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[:, lower_joint_index]) * 100
        error_dict['L-PE'].append(lower_error_)


        avg_error = {k: float((sum(v) / len(v)).detach().cpu().numpy()) for k, v in error_dict.items()}

        task = opt['task']
        info = f'[{task}][epoch-{epoch}]: '
        for k, v in avg_error.items():
            info += str(k) + ':' + str(v) + ', '

        # testing log
        logger.info(info)
    return avg_error

def main(opt, save_animation=False):
    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

    option.save(opt)
    opt = option.dict_to_nonedict(opt)

    logger_name = 'test'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        elif phase == 'train':
            continue
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)


    model = define_Model(opt)
    model.init_test()

    _ = evaluate(opt, logger, model, test_loader, save_animation=save_animation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/opt_ajlm.json', help='Path to option JSON file.')
    parser.add_argument('--task', type=str, default='AvatarJLM', help='Experiment name.')
    parser.add_argument('--protocol', type=str, choices=['1', '2', '3', 'real'], required=True, help='Protocol.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Trained model weights.')
    parser.add_argument('--vis', action="store_true", help='Save animation.')
    args = parser.parse_args()
    opt = option.parse(args.opt, args, is_train=False)
    main(opt, args.vis)