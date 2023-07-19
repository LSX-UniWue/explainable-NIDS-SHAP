
import numpy as np
import pandas as pd


if __name__ == '__main__':
    summary = 'cidds'

    summary_path = f'./{summary}_summary.csv'
    df = pd.read_csv(summary_path, header=0)
    df.iloc[:, 2:] = df.iloc[:, 2:] * 100
    df = df.round(decimals=1)
    cols = [df.iloc[:, 0], df.iloc[:, 1]]  # first two cols
    for i in range(3, df.shape[-1], 2):
        cols.append('$' + pd.Series(df.iloc[:, i - 1].astype(str) + ' \pm ' + df.iloc[:, i].astype(str) + '$',
                              name=df.iloc[:, i - 1].name))
    df = pd.concat(cols, axis=1)
    # df = df.sort_values(list(df.columns[:2].values))
    print(df.to_latex(index=False, escape=False))
