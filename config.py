import os
import sys
import time
import torch
import random
import logging
import argparse
from datetime import datetime


def get_config():
    parser = argparse.ArgumentParser()

    '''Base'''
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='chinese_roberta')
    parser.add_argument('--method_name', type=str, default='bilstm')

    '''File'''
    parser.add_argument('-l', '--language', type=str, default='cn')
    parser.add_argument('-m', '--model_type', type=str, default='cf')
    parser.add_argument('--train_file', type=str, default='big.csv')
    parser.add_argument('-t', '--test_file', type=str, default='ccft_cn.csv')

    '''Optimization'''
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    '''Environment'''
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))

    args = parser.parse_args()
    args.device = torch.device(args.device)
    if args.language == 'cn':
        args.model_name = 'chinese_roberta'
    else:
        args.model_name = 'roberta'

    '''logger'''
    args.log_name = '{}_{}_{}_{}_{}.log'.format(args.language, args.model_type, args.model_name, args.method_name,
                                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args, logger
