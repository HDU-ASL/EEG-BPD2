# Imports
import numpy as np
from random import random
from config import get_config, activation_dict
from data_loader import get_loader
#from solver import Solver

from solver_maps import Solver

import torch


if __name__ == '__main__':
    
    # Setting random seed
    random_name = str(random())
    random_seed = 42 #336   
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    # Task: visual classification, facial emotion recognition
    task = 'facial' # 'facial' or 'visual'
    
    # Setting the config for each stage
    train_config = get_config(mode='train', task=task)
    dev_config = get_config(mode='val', task=task)
    test_config = get_config(mode='test', task=task)

    #print(train_config)

    # Creating pytorch dataloaders
    train_data_loader,test_data_loader = get_loader(train_config)
    dev_data_loader = test_data_loader

    # Solver is a wrapper for model training and testing
    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)

    # # Build the model
    solver.build()

    solver.eval(mode="dev",to_print=True)
    




