import argparse
import logging
import sys
from datetime import date
import random

import torch
from torch import nn
from torch.optim import Adam, AdamW
from trainers.cmgp_trainer import CMGPTrainer
from cmgp.datasets import load
from cmgp.utils.metrics import sqrt_PEHE_with_diff

arg_trainer_map = {'gpytorch_multitask_ihdp_trainer': CMGPTrainer}
arg_optimizer_map = {'adamW': AdamW, 'adam': Adam}


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run ITE estimation experiments with GPs')

    parser.add_argument('--epochs',
                        default=50,
                        type=int,
                        help='number of epochs to train model')
    parser.add_argument('--device',
                        '-d',
                        default='cpu',
                        type=str,
                        help='cpu or gpu ID to use')
    parser.add_argument('--batch_size',
                        default=50000,
                        type=int,
                        help='mini-batch size used to train model')
    parser.add_argument('--dropout_prob',
                        default=0.3,
                        type=float,
                        help='probability for dropout layers')
    parser.add_argument('--save_dir', help='path to saved model files')
    parser.add_argument('--data_dir', help='path to data files')
    parser.add_argument('--optimizer',
                        default='adamW',
                        help='type of optimizer to use')
    parser.add_argument('--num_repeats',
                        default=3,
                        type=int,
                        help='number of times to repeat experiment')
    parser.add_argument('--seed',
                        default=11202022,
                        type=int,
                        help='random seed to be used in numpy and torch')
    parser.add_argument('--learning_rate',
                        default=1e-3,
                        type=float,
                        help='learning rate for optimizer')
    parser.add_argument('--hidden_size',
                        default=512,
                        type=int,
                        help='dimensionality of hidden layers')
    parser.add_argument('--trainer_type', help='type of experiment to run')

    args = parser.parse_args()
    configs = args.__dict__

    # need this precision for GP fitting
    torch.set_default_dtype(torch.float64)

    # for repeatability
    torch.manual_seed(configs['seed'])
    random.seed(configs['seed'])

    # set up logging
    filename = f'{configs["trainer_type"]}-{date.today()}'
    FORMAT = '%(asctime)s;%(levelname)s;%(message)s'
    logging.basicConfig(level=logging.INFO,
                        filename=f'{configs["save_dir"]}logs/{filename}.log',
                        filemode='a',
                        format=FORMAT)
    logging.info(configs)

    # get trainer
    # trainer_type = arg_trainer_map[configs['trainer_type']]
    # trainer = trainer_type(
    #     optimizer_type=arg_optimizer_map[configs['optimizer']],
    #     criterion=nn.CrossEntropyLoss(reduction='sum'),
    #     **configs)

    # load data
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("ihdp")

    # perform experiment n times
    for iter in range(configs['num_repeats']):
        trainer = CMGPTrainer(X_train, W_train, Y_train)
        pred = trainer.predict(X_train)
        pehe = sqrt_PEHE_with_diff(Y_train_full, pred)

        print(f"in sample PEHE score for CMGP on ihdp = {pehe}")
        pred = trainer.predict(X_test)

        pehe = sqrt_PEHE_with_diff(Y_test, pred)

        print(f"out of sample PEHE score for CMGP on ihdp = {pehe}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
