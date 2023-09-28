import os.path
import math
import argparse
import random
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from test import evaluate


def main(opt):
    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

    if opt['datasets']['train']['resume']:
        init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
        opt['path']['pretrained_netG'] = init_path_G
        current_step = init_iter
    else:
        current_step = 0

    option.save(opt)
    opt = option.dict_to_nonedict(opt)

    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)


    model = define_Model(opt)
    model.init_train()

    eval_results = list()

    for epoch in range(opt['train']['total_step']):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1

            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            model.update_learning_rate(current_step)

            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            if current_step % opt['train']['checkpoint_test'] == 0:
                avg_error = evaluate(opt, logger, model, test_loader, epoch)
                # top-3
                eval_results.append(list(avg_error.values()) + [current_step])
                eval_results.sort()
                logger.info(eval_results[:3])

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/opt_ajlm.json', help='Path to option JSON file.')
    parser.add_argument('--task', type=str, default='AvatarJLM', help='Experiment name.')
    parser.add_argument('--protocol', type=str, choices=['1', '2', '3'], required=True, help='Protocol.')
    args = parser.parse_args()
    opt = option.parse(args.opt, args, is_train=True)
    main(opt)