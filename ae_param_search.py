
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ae_cidds_run import detect_anomalies


def get_param_grid(cpus):
    return ParameterGrid({'cpus': [cpus],
                          'n_layers': [3],
                          'n_bottleneck': [32],
                          'epochs': [10],
                          'batch_size': [2048],
                          'learning_rate': [1e-2],
                          'verbose': [2],
                          'device': ['cuda'],
                          'seed': [0]})
    # return ParameterGrid({'cpus': [cpus],
    #                       'n_layers': [2, 3, 4],
    #                       'n_bottleneck': [8, 16, 32],
    #                       'epochs': [10],
    #                       'batch_size': [2048],
    #                       'learning_rate': [1e-2, 1e-3, 1e-4],
    #                       'verbose': [2],
    #                       'device': ['cuda'],
    #                       'seed': list(range(10))})


if __name__ == '__main__':

    evaluation_save_path = 'outputs/models/cidds/logs'
    job_name = 'ae_local_0'  # save model under f'./outputs/models/cidds/{job_name}_state_dict.pt', use None to not save
    cpus = 8

    for param_dict in get_param_grid(cpus):
        print(datetime.now())
        seed = param_dict['seed']
        detect_anomalies(params=param_dict,
                         seed=seed,
                         evaluation_save_path=evaluation_save_path,
                         job_name=job_name)
        print(datetime.now())
