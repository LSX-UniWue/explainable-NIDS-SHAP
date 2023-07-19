
from pathlib import Path
from datetime import datetime

import pandas as pd


print('Start time:', datetime.now())
data_path = '../data_raw'
# Load data
data = pd.read_csv(Path(data_path) / 'all4weeks.csv.gz', compression='gzip', header=0)
data.columns = data.columns.str.strip()
data['Date first seen'] = pd.to_datetime(data['Date first seen'])
data[['dummy']] = 1


def get_num_flows_in_time_window(df, window='10s'):
    df = df.sort_values('Date first seen')
    return df.rolling(window, on='Date first seen')['dummy'].sum()


dst_traffic = data[['Dst IP Addr', 'Dst Pt', 'Date first seen', 'dummy']].groupby(['Dst IP Addr', 'Dst Pt'])
dst_traffic = dst_traffic.apply(get_num_flows_in_time_window).reset_index().rename({'dummy': 'Dst Connections last 10s', 'level_2': 'iloc'}, axis=1)
data = pd.merge(left=data, right=dst_traffic[['iloc', 'Dst Connections last 10s']], how='left', left_index=True, right_on='iloc').drop('iloc', axis=1)

src_traffic = data[['Src IP Addr', 'Src Pt', 'Date first seen', 'dummy']].groupby(['Src IP Addr', 'Src Pt'])
src_traffic = src_traffic.apply(get_num_flows_in_time_window).reset_index().rename({'dummy': 'Src Connections last 10s', 'level_2': 'iloc'}, axis=1)
data = pd.merge(left=data, right=src_traffic[['iloc', 'Src Connections last 10s']], how='left', left_index=True, right_on='iloc').drop('iloc', axis=1)
data = data.drop('dummy', axis=1).sort_values('Date first seen')

data.to_csv(Path(data_path) / 'all4weeks_extended.csv', index=False)

print('End time:', datetime.now())
