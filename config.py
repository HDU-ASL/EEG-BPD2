# Imports
import argparse
from datetime import datetime
from datetime import datetime
import pprint
from torch import optim
import torch.nn as nn


# choices for optimizer and activation
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}


def str2bool(v):
    # string to boolean
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        # Configuration Class: set kwargs as class attributes with setattr
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)

    def __str__(self):
        # Pretty-print configurations in alphabetical order
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, task=None, **optional_kwargs):
    
    # Get configurations as attributes of class
    # 1. Parse configurations with argparse.
    # 2. Create Config class initilized with parsed kwargs.
    # 3. Return Config class.

    # Decided by task
    if task == 'facial' :
        data_opt = 'facial'
        class_opt = 6
        batch_opt = 32 # 16
        epoch_opt = 3000 # 100
        lr_opt = 0.0005
        diff_weight_opt = 0.5
        sim_weight_opt = 0.5
        requires_opt = True


    parser = argparse.ArgumentParser()
    parser.add_argument('-sub','--subject', default=0   , type=int, help="choose a subject from 1 to 6, default is 0 (all subjects)")
    parser.add_argument('-tl', '--time_low', default=20, type=float, help="lowest time value")
    parser.add_argument('-th', '--time_high', default=460,  type=float, help="highest time value")
    # Facial_path
    
    parser.add_argument('-fed', '--facial-eeg-dataset', default=r"/root/autodl-tmp/test/BCML/data/Work/EEG/", help="Image dataset path")
    parser.add_argument('-fid', '--facial-image-dataset', default=r"/root/autodl-tmp/test/BCML/data/Work/IMAGE/", help="Image dataset path")
    parser.add_argument('-fsp', '--facial-splits-path', default=r"/home/hyx/test/BCML/data/Work/SPLIT/", help="Image dataset path")

    # Data options
    parser.add_argument('--data', type=str, default=data_opt)
    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--use_sim', type=str2bool, default=True)
    parser.add_argument('--requires_grad', type=str2bool, default=requires_opt)
    #parser.add_argument('--feature_extractor')

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--num_classes', type=int, default=class_opt)
    parser.add_argument('--batch_size', type=int, default=batch_opt)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=epoch_opt)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--num_trials', type=int, default=2)



    parser.add_argument('--sax', type=float, default=-3.0)
    parser.add_argument('--saq', type=float, default=-3.0)
    parser.add_argument('--sax1', type=float, default=-3.0)
    parser.add_argument('--saq1', type=float, default=0.0)
    parser.add_argument('--sax2', type=float, default=-3.0)
    parser.add_argument('--saq2', type=float, default=0.0)   
    parser.add_argument('--cla_weight', type=float, default=1.0)
    parser.add_argument('--diff_weight', type=float, default=diff_weight_opt)
    parser.add_argument('--sim_weight', type=float, default=sim_weight_opt)
    parser.add_argument('--sp_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=0.0)

    parser.add_argument('--learning_rate', type=float, default=lr_opt)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)

    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='relu')


    # Model
    parser.add_argument('--model', type=str,
                        default='BMCL', help='one of {BMCL, }')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
