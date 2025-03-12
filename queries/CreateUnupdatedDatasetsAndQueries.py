import pickle as pkl
from typing import Dict
import os
import pandas as pd
import numpy as np
import common
from datasets import JoinOrderBenchmark
from queries.ConvertMSCNTestWorkload import MakeIMDBTables
import numba

np.random.seed(42)
use_cols = None
sample_rate = 0.8

tables_name = [name.replace('.csv', '') for name in JoinOrderBenchmark.GetJobLightJoinKeys().keys()]
tables_dict: Dict[str, common.CsvTable] = MakeIMDBTables(tables_name, data_dir='../datasets/job/', use_cols=use_cols)

@numba.jit(nopython=True)
def index_csv(lines, sample_index):
    new_lines = [lines[0]]  # header
    new_lines.extend([lines[i + 1] for i in sample_index])  # skip header by +1
    return new_lines

for name, table in tables_dict.items():
    df = pd.read_csv(os.path.join('../datasets/job/', name + '.csv'),  usecols=use_cols,  escapechar='\\', low_memory=False)
    sample_index = np.random.choice(np.arange(df.shape[0]), int(df.shape[0]*sample_rate), replace=False)
    with open(f'../datasets/job/{name}_previous_index.pkl', 'wb') as f:
        pkl.dump(sample_index, f)
    with open(os.path.join('../datasets/job/', name + '.csv'), 'r') as f:
        lines = f.readlines()

    new_lines = index_csv(lines, sample_index)
    with open(os.path.join('../datasets/job/', name + '_previous.csv'), 'w') as f:
        f.writelines(new_lines)
    # 始终无法使用pandas按IMDB标准格式转义字符串，直接
    # df.iloc[sample_index].to_csv(f'../datasets/job/{name}_previous.csv', escapechar='\\',  quotechar='"', index=False,
    #                              # we know that IMDB doesn't contain float type
    #                              float_format='%d'
    #                              )