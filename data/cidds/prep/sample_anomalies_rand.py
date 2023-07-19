
from pathlib import Path
import numpy as np
import pandas as pd

from data.cidds.util import get_summed_columns, map_orig_columns, get_cols_and_dtypes

# Sample anomalies for each attack type
# load data
cols, dtypes = get_cols_and_dtypes(cat_encoding='onehot', num_encoding='quantized')
data = pd.read_csv(Path('./') / 'data' / 'cidds' / 'data_prep' / 'onehot_quantized' / f'test.csv.gz',
                   index_col=None, usecols=cols + ['attackType', 'Date first seen'],
                   dtype={'attackType': str, 'Date first seen': str, **dtypes}, header=0, compression='gzip')

attacks = []
for attack_type in ['dos', 'portScan', 'pingScan', 'bruteForce']:
    att_data = data[data['attackType'] == attack_type]
    # grab 20 random anomalies
    att_data = att_data.sample(n=20)
    attacks.append(att_data)

attacks = pd.concat(attacks)
attacks.to_csv(Path('../') / 'data_prep' / 'onehot_quantized' / f'anoms.csv')

# Grab unencoded anomaly values
test = pd.read_csv(Path('../') / 'data_raw' / 'test_unencoded.csv', index_col=None)
anom_rand = pd.read_csv(Path('../') / 'data_prep' / 'onehot_quantized' / 'anoms.csv', index_col=0)
slice = test.loc[anom_rand.index]
slice.to_csv(Path('../') / 'data_raw' / 'anoms.csv')

# extend and order in line with our preprocessing
data = pd.read_csv(Path('../') / 'data_raw' / f'anoms.csv', index_col=0)
data.columns = map_orig_columns(data.columns)
# split date into day and time
data['Date'] = pd.to_datetime(data['Date'])
data['Weekday'] = data['Date'].dt.dayofweek
data['Daytime'] = data['Date'].dt.time
data = data.drop(columns=['Date'])
# split flags into individual columns
data['SYN'] = data['Flags'].str.contains('S').astype(int)
data['ACK'] = data['Flags'].str.contains('A').astype(int)
data['FIN'] = data['Flags'].str.contains('F').astype(int)
data['PSH'] = data['Flags'].str.contains('P').astype(int)
data['URG'] = data['Flags'].str.contains('U').astype(int)
data['RES'] = data['Flags'].str.contains('R').astype(int)
data = data.drop(columns=['Flags'])

data = data[get_summed_columns() + ['attackType']]

# save data
data.to_csv(Path('../') / 'data_raw' / 'anoms_sorted.csv')
