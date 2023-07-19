
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from oc_cidds_run import detect_anomalies


def get_param_grid():

    return ParameterGrid({"kernel": ['rbf'],
                          'gamma': [1e1],
                          'tol': [1e-3],
                          'nu': [0.2],
                          'shrinking': [1],
                          'cache_size': [500],
                          'max_iter': [-1],
                          'seed': [0]})
    # return ParameterGrid({"kernel": ['rbf'],
    #                       'gamma': [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3],
    #                       'tol': [1e-3],
    #                       'nu': [0.2, 0.4, 0.6, 0.8],
    #                       'shrinking': [1],
    #                       'cache_size': [500],
    #                       'max_iter': [-1],
    #                       'seed': list(range(10))})


if __name__ == '__main__':

    evaluation_save_path = 'outputs/models/cidds/logs'
    job_name = 'oc_local_0'

    for param_dict in get_param_grid():
        print(datetime.now())
        seed = param_dict.pop('seed')
        detect_anomalies(params=param_dict,
                         seed=seed,
                         evaluation_save_path=evaluation_save_path,
                         job_name=job_name)
        print(datetime.now())
