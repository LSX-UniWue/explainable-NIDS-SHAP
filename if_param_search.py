
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from if_cidds_run import detect_anomalies


def get_param_grid():
    return ParameterGrid({'n_estimators': [32],
                          'max_samples': [0.8],
                          'max_features': [0.8],
                          'bootstrap': [0],
                          'n_jobs': [-1],
                          'random_state': [0]})
    # return ParameterGrid({'n_estimators': [2 ** n for n in range(4, 8)],
    #                       'max_samples': [0.4, 0.6, 0.8, 1.],
    #                       'max_features': [0.4, 0.6, 0.8],
    #                       'bootstrap': [0],
    #                       'n_jobs': [-1],
    #                       'random_state': list(range(10))})


if __name__ == '__main__':

    evaluation_save_path = 'outputs/models/cidds/logs'
    job_name = 'if_local_0'  # used to save model under f'./outputs/models/cidds/{job_name}.pkl', use None to not save

    for param_dict in get_param_grid():
        print(datetime.now())
        seed = param_dict['random_state']
        detect_anomalies(params=param_dict,
                         seed=seed,
                         evaluation_save_path=evaluation_save_path,
                         job_name=job_name)
        print(datetime.now())
