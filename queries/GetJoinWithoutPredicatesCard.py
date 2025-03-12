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

seed = 42
np.random.seed(seed)
hows = AQP_estimator.OPS_array.tolist()
use_cols = 'general'
test_workloads = ['job-light', 'job-light-ranges']
shorten_table_name = {
    'title': 't',
    'cast_info': 'ci',
    'movie_info': 'mi',
    'movie_keyword': 'mk',
    'movie_companies': 'mc',
    'movie_info_idx': 'mi_idx',
}
tags = ['','_previous']

def MakeIMDBTables(dataset, data_dir, use_cols, tag=''):
    return datasets.LoadImdb(dataset, data_dir=data_dir, use_cols=use_cols, try_load_parsed=False, tag=tag)


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
        tables_name = [name for name in JoinOrderBenchmark.GetJobLightJoinKeys().keys()]
        tables_dict: Dict[str, common.CsvTable] = MakeIMDBTables(tables_name, data_dir='../datasets/job/', use_cols=use_cols, tag=tag)
        # if tag != '':
        #     for tab_name in tables_name:
        #         tables_dict[tab_name.replace(tag, '')] = tables_dict[tab_name]
        print(tables_name)
        # tables_dict = MakeIMDBTables(tables_name, data_dir='../datasets/job/', use_cols='full')
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
        true_cards = {how: {} for how in hows}
        for workload in test_workloads:
            for how in hows:
                print('converting '+workload+'-'+how)
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
    
                workload_num = len(loaded_queries)
                for query_id, (join_tables, join_keys, preds, true_card) in tqdm(enumerate(loaded_queries)):
                    print(f'query_id: {query_id}\n join_tables: {join_tables}, join_keys:{join_keys}, preds:{preds}')
                    predicates = []
                    used = []
                    for table_name in join_tables:
                        predicates.append(AQP_estimator.DirectEstimator.GetKeyPredicate(table_name, AQP_estimator.DirectEstimator.GetJoinKeyColumn(tables_dict[table_name])))
                    # get true card
                    condition_prob = estimator.get_prob_of_predicate_tree(predicates, join_tables, tables_dict, how, real=True)
                    true_card = int(condition_prob)
                    true_cards[how][frozenset(join_tables)]=true_card
    
        with open(f'basecard{tag}.pkl', 'wb') as f:
            pkl.dump(true_cards, f)
