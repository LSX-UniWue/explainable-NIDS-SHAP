
import glob
import numpy as np
import pandas as pd


if __name__ == '__main__':
    """Join logs from multiple runs into one file."""
    files = []
    for file_path in glob.glob('./OneClassSVM/*.csv'):
        files.append(pd.read_csv(file_path))
    df = pd.concat(files, ignore_index=True)
    df.to_csv('summary_oc_.csv', index=False)
