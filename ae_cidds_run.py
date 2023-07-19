
import datetime
from pathlib import Path
from collections import ChainMap
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch.utils.data

from anomaly_detection.autoencoder_torch import Autoencoder
from data.cidds.util import get_cols_and_dtypes


def detect_anomalies(params, seed, evaluation_save_path, model_load_path=None, job_name=None, device='cpu'):
    # setup args
    source_path = './'
    num_encoding = 'quantized'
    data_folder = 'onehot_quantized'
    device = params.pop('device')  # 'cuda', 'cpu'
    print(f'device: {device}')

    np.random.seed(seed)

    # Load & train model
    cols, dtypes = get_cols_and_dtypes(cat_encoding='onehot', num_encoding=num_encoding)
    params['n_inputs'] = len(cols)
    detector = Autoencoder(**params)
    if model_load_path is not None:
        detector.load(model_load_path)
    else:
        # Load data and train
        train = pd.read_csv(Path(source_path) / 'data' / 'cidds' / 'data_prep' / data_folder / 'train.csv.gz',
                            index_col=None, usecols=cols, header=0, dtype=dtypes, compression='gzip')[cols]
        x = torch.Tensor(train.values)
        y = torch.zeros((train.values.shape[0], 1))
        train_dataset = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['batch_size'],
                                                   num_workers=0, shuffle=True)
        detector = detector.fit(train_loader, device=device)

        if job_name is not None:
            detector.save(f'./outputs/models/cidds/ae_{job_name}_state_dict.pt')

    # Evaluation
    eval_out = []
    for split in ['valid', 'test']:
        eval_data = pd.read_csv(Path(source_path) / 'data' / 'cidds' / 'data_prep' / data_folder / f'{split}.csv.gz',
                                index_col=None, usecols=cols + ['isNormal'], header=0,
                                dtype={'isNormal': np.int8, **dtypes}, compression='gzip')[cols + ['isNormal']]
        y_eval = torch.tensor(1 - eval_data['isNormal'])
        eval_data = eval_data.drop(['isNormal'], axis=1)
        x = torch.Tensor(eval_data[cols].values)
        eval_data = torch.utils.data.TensorDataset(x, y_eval)
        eval_loader = torch.utils.data.DataLoader(dataset=eval_data, batch_size=params['batch_size'],
                                                  num_workers=0, shuffle=False)
        out_dict = detector.test(eval_loader, device=device)
        out_dict = {key + f'_{split}': val for key, val in out_dict.items()}
        print(out_dict)
        eval_out.append(out_dict)

    # Outputs
    if evaluation_save_path is not None:
        out_dict = dict(ChainMap(*eval_out))
        out_df = pd.DataFrame()
        out_df = out_df.append({**params, **out_dict, 'seed': seed},
                               ignore_index=True)
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        out_df.to_csv(Path(evaluation_save_path) / 'Autoencoder' / f'{curr_time}.csv', index=False)


if __name__ == '__main__':
    """
    Argparser needs to accept all possible param_search arguments, but only passes given args to params.
    """
    str_args = ('model_load_path', 'evaluation_save_path', 'device', 'job_name')
    float_args = ['learning_rate']
    int_args = ('cpus', 'n_layers', 'n_bottleneck', 'epochs', 'batch_size', 'verbose', 'seed')
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
    model_load_path = args_dict.pop('model_load_path')
    job_name = args_dict.pop('job_name')
    params = {key: val for key, val in args_dict.items() if val}  # remove entries with None values

    detect_anomalies(params=params,
                     seed=args_dict['seed'],
                     evaluation_save_path=evaluation_save_path,
                     model_load_path=model_load_path,
                     job_name=job_name)
