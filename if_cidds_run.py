
import datetime
from pathlib import Path
from collections import ChainMap
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from data.cidds.util import get_cols_and_dtypes


def detect_anomalies(params, seed, evaluation_save_path, job_name=None):
    np.random.seed(seed)

    source_path = './'
    cols, dtypes = get_cols_and_dtypes(cat_encoding='onehot', num_encoding='quantized')

    # Set detector
    detector_class = IsolationForest
    params['random_state'] = seed

    # Training
    print(f'Loading train ...')
    train = pd.read_csv(Path(source_path) / 'data' / 'cidds' / 'data_prep' / 'onehot_quantized' / 'train.csv.gz',
                        index_col=None, usecols=cols, header=0, dtype=dtypes, compression='gzip')[cols]

    print('Training ...')
    detector = detector_class(**params).fit(train)
    del train

    # Anomaly classification outputs
    eval_out = []
    for split in ['valid', 'test']:
        print(f'Loading {split} ...')
        eval_data = pd.read_csv(Path(source_path) / 'data' / 'cidds' / 'data_prep' / 'onehot_quantized' / f'{split}.csv.gz',
                                index_col=None, usecols=cols + ['isNormal'], header=0,
                                dtype={'isNormal': np.int8, **dtypes}, compression='gzip')[cols + ['isNormal']]
        y = 1 - eval_data.pop('isNormal')

        print(f'Evaluating {split} ...')
        # invert scores so that the higher the more anomalous
        scores = -1 * pd.Series(detector.score_samples(eval_data), index=eval_data.index)
        del eval_data

        out_dict = {f'auc_pr_{split}': average_precision_score(y_true=y, y_score=scores),
                    f'auc_roc_{split}': roc_auc_score(y_true=y, y_score=scores)}
        print(out_dict)
        eval_out.append(out_dict)

    out_dict = dict(ChainMap(*eval_out))
    out_df = pd.DataFrame()
    out_df = out_df.append({**params, **out_dict, 'seed': seed, 'job_name': job_name}, ignore_index=True)
    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    out_path = Path(evaluation_save_path) / 'IsolationForest'
    if not out_path.exists():
        out_path.mkdir(parents=True)
    out_df.to_csv(out_path / f'{curr_time}.csv', index=False)

    if job_name is not None:
        import joblib
        joblib.dump(detector, f'./outputs/models/cidds/{job_name}.pkl')


if __name__ == '__main__':

    str_args = ('evaluation_save_path', 'job_name')
    float_args = ('max_samples', 'max_features')
    int_args = ('n_estimators', 'random_state', 'bootstrap', 'n_jobs')
    bool_args = ()
    parser = ArgumentParser()
    for arg in str_args:
        parser.add_argument(f'--{arg}')
    for arg in int_args:
        parser.add_argument(f'--{arg}', type=int)
    for arg in float_args:
        parser.add_argument(f'--{arg}', type=float)
    for arg in bool_args:
        parser.add_argument(f'--{arg}', action='store_true')
    args_dict = vars(parser.parse_args())

    evaluation_save_path = args_dict.pop('evaluation_save_path')
    job_name = args_dict.pop('job_name')

    detect_anomalies(params=args_dict,
                     seed=args_dict['random_state'],
                     evaluation_save_path=evaluation_save_path,
                     job_name=job_name)
