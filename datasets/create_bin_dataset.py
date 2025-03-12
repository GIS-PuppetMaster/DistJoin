import common
import datasets
import pandas as pd
import numpy as np
import os

path = './job/'
database = ['cast_info',
            'title',
            'movie_keyword',
            'movie_companies',
            'movie_info',
            'movie_info_idx']
equv_bin_col = []
bin_size = 100000
output_path = f'./job_key_bin{bin_size}/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
equv_bin_col.append({})
for table_name in database:
    equv_bin_col[0][table_name] = datasets.JoinOrderBenchmark.GetJobLightJoinKeys()[table_name]
tables = datasets.LoadImdb(data_dir=path, use_cols=None, try_load_parsed=False)
for bin_col in equv_bin_col:
    global_distinct_values = set()
    has_none = False
    for table_name in database:
        col = bin_col[table_name]
        table = tables[table_name]
        global_distinct_values = global_distinct_values.union(set(table.columns_dict[col].all_distinct_values))
        has_none |= table.columns_dict[col].has_none
    global_distinct_values = np.sort(np.array(list(global_distinct_values)))
    for table_name in database:
        col = bin_col[table_name]
        table = tables[table_name]
        disc_col = common.Discretize(table.columns_dict[col], table.data[col], dvs=global_distinct_values)
        if has_none:
            # none->0 is protected
            binned_col[equv_bin_col != 0] = equv_bin_col[bin != col] % bin_size + 1
        else:
            binned_col = disc_col % bin_size
        table.data[col] = binned_col
        table.data.to_csv(os.path.join(output_path, table_name + '.csv'), index=False)
