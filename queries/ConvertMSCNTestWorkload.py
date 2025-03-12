import csv
from typing import Dict

import numpy as np
import pandas as pd
import os
import pickle as pkl
from tqdm import tqdm

import AQP_estimator
from AQP_estimator import DirectEstimator
import utils.util as util
import torch

import common
import datasets
from datasets import JoinOrderBenchmark
from utils.util import MakeIMDBTables
from convert_util import *
seed = 42
np.random.seed(seed)
hows = AQP_estimator.OPS_array.tolist()
tags = ['','_previous']
sample_num=10000
use_cols = None
test_workloads = ['job-light', 'job-light-ranges']
shorten_table_name = {
    'title': 't',
    'cast_info': 'ci',
    'movie_info': 'mi',
    'movie_keyword': 'mk',
    'movie_companies': 'mc',
    'movie_info_idx': 'mi_idx',
}




OPS = [torch.eq, torch.greater, torch.less, torch.greater_equal, torch.less_equal]


def convert_with_pandas_numpy(table: common.CsvTable):
    """使用 Pandas 和 NumPy 高效编码表格"""
    df = table.data
    encoded_df = pd.DataFrame(index=df.index)  # 初始化编码表格
    for col in df.columns:
        # 创建值到索引的映射
        distinct_values = np.array(table.columns_dict[col].all_distinct_values)
        value_to_index = {v: i for i, v in enumerate(distinct_values)}

        # 使用 Pandas map 矢量化编码
        encoded_df[col] = df[col].map(value_to_index).astype('int32')

    return encoded_df


if __name__ == '__main__':
    os.makedirs('./MSCN/', exist_ok=True)
    for tag in tags:
        tables_name = [name.replace('.csv', tag) for name in JoinOrderBenchmark.GetJobLightJoinKeys().keys()]
        tables_dict: Dict[str, common.CsvTable] = MakeIMDBTables(tables_name, data_dir='../datasets/job/', use_cols=use_cols, tag=tag)
        print(tables_name)
        tables = [tables_dict[name] for name in tables_name]  # keep order to give determined generation that can be reproduced
        global_distinct_values = set()
        for dataset in tables_name:
            table = tables_dict[dataset]
            key_col = table.columns_dict[JoinOrderBenchmark.GetJobLightJoinKeys()[dataset.replace(tag, '')]]
            dvs = key_col.all_distinct_values
            if key_col.has_none:
                dvs = np.delete(dvs, 0)
            global_distinct_values = global_distinct_values.union(set(dvs))
        # key为int类型
        global_distinct_values = np.sort(np.array(list(global_distinct_values)))
        if np.issubdtype(global_distinct_values.dtype, np.datetime64):
            global_distinct_values = np.insert(global_distinct_values, 0, np.datetime64('NaT'))
        else:
            if global_distinct_values.dtype != np.dtype(float) and global_distinct_values.dtype != np.dtype(object):
                global_distinct_values = global_distinct_values.astype(float)
            global_distinct_values = np.insert(global_distinct_values, 0, np.nan)
        for dataset in tables_name:
            table = tables_dict[dataset]
            key_col = table.columns_dict[JoinOrderBenchmark.GetJobLightJoinKeys()[dataset.replace(tag, '')]]
            key_col.SetGlobalDiscretizeMask(global_distinct_values)
        # global_distinct_values = torch.as_tensor(global_distinct_values, device=util.get_device())
        estimator = DirectEstimator(None,
                                    tables_dict,
                                    tables_dict,
                                    global_distinct_values=global_distinct_values,
                                    config={'test': {'faster_version': True}},
                                    device=util.get_device())
        # convert table to numeric
        for table in tqdm(tables, desc="discretion data"):
            if not os.path.exists(f'./MSCN/{table.name}_numeric{tag}.pkl') or not os.path.exists(f'./MSCN/{table.name}_numeric{tag}.csv') :
                # numeric_table = convert_table_to_numeric(table)
                numeric_table = convert_with_pandas_numpy(table)
                numeric_table.to_csv(f'./MSCN/{table.name}_numeric{tag}.csv', index=False)
                with open(f'./MSCN/{table.name}_numeric{tag}.pkl', 'wb') as f:
                    pkl.dump(numeric_table, f)

        # generating column_min_max_vals.csv
        if not os.path.exists(f'./MSCN/column_min_max_vals{tag}.csv'):
            column_min_max_vals = {'name':[],
                                   'min':[],
                                   'max':[],
                                   'cardinality':[],
                                   'num_unique_values':[]}
            for table in tqdm(tables, desc="generating column_min_max_vals.csv"):
                t_name = shorten_table_name[table.name]
                for col in table.columns:
                    column_min_max_vals['name'].append(f'{t_name}.{col.name}')
                    column_min_max_vals['min'].append(0)
                    column_min_max_vals['max'].append(len(col.all_distinct_values)-1)
                    column_min_max_vals['cardinality'].append(table.data.shape[0])
                    column_min_max_vals['num_unique_values'].append(len(col.all_distinct_values))
            column_min_max_vals = pd.DataFrame(column_min_max_vals)
            column_min_max_vals.to_csv(f'./MSCN/column_min_max_vals{tag}.csv', index=False)

        for workload in test_workloads:
            for how in hows:
                print('converting '+workload+'-'+how)
                # generating samples for bitmaps
                for table in tables:
                    if not os.path.exists(f'./MSCN/{table.name}_sample{sample_num}_{use_cols}{tag}.pkl'):
                        table.sample = table.data.sample(sample_num, replace=False)
                        with open(f'./MSCN/{table.name}_sample{sample_num}_{use_cols}{tag}.pkl', 'wb') as f:
                            pkl.dump(table.sample, f)
                    else:
                        with open(f'./MSCN/{table.name}_sample{sample_num}_{use_cols}{tag}.pkl', 'rb') as f:
                            table.sample = pkl.load(f)
                # how = how
                join_queries = []
                raw_join_queries = []
                join_bitmaps = []

                # this must be run on python3.7+ since the dict is in the order of inserting now, otherwise the samples' order would be incorrect
                queries_job_format = util.JobToQuery(workload+'.csv')
                queries_job_format_ = []
                for idx, (involved_tables, _, _, _) in enumerate(queries_job_format):
                    if len(set(JoinOrderBenchmark.GetJobLightJoinKeys().keys()).intersection(set(involved_tables))) < len(
                            involved_tables):
                        print('skip query since some tables are not supported')
                        continue
                    else:
                        queries_job_format_.append(queries_job_format[idx])
                queries_job_format = queries_job_format_
                loaded_queries = util.UnpackQueries(estimator.tables_dict, queries_job_format)
                total_avg_num = total_sum_num = total_max_num = total_min_num = num = len(loaded_queries)
                result_list = []
                if os.path.exists(f'{workload}_{how}{tag}.pkl'):
                    with open(f'{workload}_{how}{tag}.pkl', 'rb') as f:
                        true_card_list = pkl.load(f)
                else:
                    true_card_list = []
                workload_num = len(loaded_queries)
                for query_id, (join_tables, join_keys, preds, true_card) in tqdm(enumerate(loaded_queries)):
                    true_card = true_card_list[query_id] if query_id < len(true_card_list) else None # otherwise means that the program is generating true cards
                    # print(f'replace true card with join how:{how}')
                    print(f'query_id: {query_id}\n join_tables: {join_tables}, join_keys:{join_keys}, preds:{preds}')
                    predicates = []
                    joined_tables = set(join_tables)
                    used_tables = set()
                    bitmaps = [np.ones(sample_num, dtype=bool) for _ in range(len(join_tables))]
                    for table_name, pred in preds.items():
                        tab_idx = join_tables.index(table_name)
                        columns = pred['cols']
                        operators = pred['ops']
                        vals = pred['vals']
                        table = estimator.fact_tables[table_name]
                        columns, operators, vals = AQP_estimator.FillInUnqueriedColumns(estimator.base_tables[table_name], columns, operators, vals)
                        # get sample's bitmap
                        bitmap = bitmaps[tab_idx]
                        for col_idx, (col, ops, vs) in enumerate(zip(columns, operators, vals)):
                            if ops is None:
                                continue
                            for op, v in zip(ops, vs):
                                if op is not None:
                                    bitmap &= AQP_estimator.OPS[op](table.sample[col.name], v)
                        bitmaps[tab_idx] = bitmap.values # make sure in correct order
                        if true_card is None:
                            for col, ops, vs in zip(columns, operators, vals):
                                if ops is None:
                                    continue
                                used_tables.add(table_name)
                                ops = list(map(lambda x: AQP_estimator.OPS_dict[x], ops))
                                vs = col.ValToBin(vs)
                                col_idx = table.base_table.columns_name_to_idx[col.name]
                                original_projected_ops = list(map(lambda x: AQP_estimator.OPS_dict[x], operators[col_idx]))
                                original_projected_vals = [col.ValToBin(x) for x in vals[col_idx]]
                                predicates.append(AQP_estimator.Predicate(table_name, col.name, table.map_from_fact_col_to_col[col.name],
                                                                          ops, vs,
                                                                          original_projected_ops, original_projected_vals,
                                                                          operators[col_idx], vals[col_idx]))

                    for table_name in join_tables:
                        predicates.append(AQP_estimator.DirectEstimator.GetKeyPredicate(table_name, AQP_estimator.DirectEstimator.GetJoinKeyColumn(tables_dict[table_name])))
                    join_bitmaps.append(np.vstack(bitmaps))
                    if true_card is None:
                        # get true card
                        condition_prob = estimator.get_prob_of_predicate_tree(predicates, join_tables, estimator.fact_tables, how, real=True)
                        true_card = int(condition_prob)
                        true_card_list.append(true_card)
                        print(f'true_card: {true_card}')

                with open(f'{workload}_{how}{tag}.pkl', 'wb') as f:
                    pkl.dump(true_card_list, f)

                print(true_card_list)
                # recreate .csv query with updated true_card
                df = pd.read_csv(f'{workload}.csv', delimiter='#',header=None)
                df[df.shape[-1] - 1] = df[df.shape[-1] - 1].astype(str)

                df.iloc[:,-1] = np.array(true_card_list, dtype=str)
                if workload == 'job-light-ranges':
                    # add shorten table names
                    tabs_queries = df.iloc[:,0].copy()
                    for k, tabs in enumerate(tabs_queries):
                        tabs = tabs.split(',')
                        tabs = [f'{tab} {shorten_table_name[tab]}' for tab in tabs]
                        tabs_queries[k] = ','.join(tabs)
                    df.iloc[:,0] = tabs_queries

                    # replace table name with shorten table name on keys
                    keys_queries = df.iloc[:,1].copy()
                    for k, keys in enumerate(keys_queries):
                        for tab_name, shorten_name in shorten_table_name.items():
                            # table.col -> shorten.col
                            keys = keys.replace(tab_name+'.', shorten_name+'.') # prevent column has the same name with table
                        keys_queries[k] = keys
                    df.iloc[:,1] = keys_queries

                    # replace table name with shorten table name on predicates
                    preds_queries = df.iloc[:, 2].copy()
                    for k, preds in enumerate(preds_queries):
                        preds = preds.split(',')
                        for ii, pred in enumerate(preds):
                            if ii%3==2:
                                # replace with bin value
                                table_col = preds[ii-2].split('.')
                                col = tables_dict[table_col[0]].columns_dict[table_col[1]]
                                preds[ii] = str(col.ValToBin(np.array(pred).astype(col.all_distinct_values.dtype).item()))
                        for ii, pred in enumerate(preds):
                            if ii%3==0:
                                for tab_name, shorten_name in shorten_table_name.items():
                                    # table.col -> shorten.col
                                    pred = pred.replace(tab_name+'.', shorten_name+'.') # prevent column has the same name with table
                                preds[ii] = pred
                        preds_queries[k] = ','.join(preds)
                    df.iloc[:, 2] = preds_queries

                # replace join how
                df.iloc[:, 1] = replace_joins(df, how)
                # df.iloc[:, 1] = df.iloc[:,1].replace('=',how, regex=True)
                df.to_csv(f'./MSCN/{workload}-{workload_num}queries-seed{seed}-{how}{tag}.csv', index=False, header=False, sep='#', quoting=csv.QUOTE_NONE)
                with open(f'./MSCN/{workload}-{workload_num}queries-seed{seed}-{how}-bitmaps{tag}.pkl', 'wb') as f:
                    pkl.dump(join_bitmaps, f)
