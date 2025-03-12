from typing import Dict, Union

import numpy as np
import pandas as pd
import os
import pickle as pkl
from tqdm import tqdm
from wandb.vendor.watchdog_0_9_0.wandb_watchdog.utils.bricks import OrderedSet

import AQP_estimator
from AQP_estimator import DirectEstimator
from utils.util import GenerateQuery, ConvertSQLQueryToBin
import utils.util as util
import torch

import common
import datasets
from datasets import JoinOrderBenchmark
from utils.util import MakeIMDBTables
seed = 42
np.random.seed(seed)
# how_flag = input('how:')
hows = AQP_estimator.OPS_array.tolist()
prob_hows = [0.1, 0.225, 0.225, 0.225, 0.225]
tags = ['', '_previous']
print(hows)
workload_num = 1000 # 50000
sample_num=1000
use_cols = 'general'

shorten_table_name = {
    'title': 't',
    'cast_info': 'ci',
    'movie_info': 'mi',
    'movie_keyword': 'mk',
    'movie_companies': 'mc',
    'movie_info_idx': 'mi_idx',
}




OPS = [torch.eq, torch.greater, torch.less, torch.greater_equal, torch.less_equal]


def convert_table_to_numeric(table):
    for c in table.columns:
        c.nan_ind = pd.isnull(c.data)
        data = util.in2d_chunked(c.data.values, c.all_distinct_values, chunk_size=10000)
        # data = util.in2d(c.data.values, c.all_distinct_values)
        if c.has_none:
            data[data == 0] = np.nan
        # c.data = torch.as_tensor(data, device=util.get_device())
    return data


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


def Query(columns, operators, vals, queue=None, return_masks=False):
    assert len(columns) == len(operators) == len(vals)

    bools = None
    for c, o, v in zip(columns, operators, vals):
        if o[0] is None:
            continue
        data = c.data
        # v = np.where(c.all_distinct_values==v)[0].item()

        inds = OPS[o[0]](data, v[0])
        if o[1] is not None:
            assert v[1] is not None
            inds = torch.bitwise_and(inds, OPS[o[1]](data, v[1]))
        if c.nan_ind.any():
            inds[c.data == 0] = False

        if bools is None:
            bools = inds
        else:
            bools &= inds
    c = bools.sum()
    if queue is not None:
        queue.put(1)
    if return_masks:
        return bools
    return c.item()


def FillInUnqueriedColumns(table, columns, operators, vals, replace_to_torch=True):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [[None, None]] * ncols, [[None, None]] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.ColumnIndex(c.name)
        os[idx] = o
        vs[idx] = torch.as_tensor(list(map(lambda x: -1 if x is None else x, v))).unsqueeze(
            0).unsqueeze(-1).pin_memory() if replace_to_torch else v
    if replace_to_torch:
        vs = torch.cat(vs, dim=-1)
    return cs, os, vs


def gen_query_single_table(dataset, table, rng, how):
    raw_query = GenerateQuery(table.columns, rng, table, dataset, how=how)
    raw_query= [[c.name for c in raw_query[0]], raw_query[1], raw_query[2]]
    return raw_query

class Predicate(AQP_estimator.Predicate):
    def __init__(self, table: str, raw_attr: Union[list, None],
                 original_predicate, original_val,
                 raw_predicate, raw_val):
        self.table = table
        self.attr = raw_attr
        self.raw_attr = raw_attr
        self.fact_predicate = None
        self.fact_val = None
        self.predicate = None
        self.val = None

        self.original_predicate = [original_predicate] if original_predicate is not None else None
        self.original_val = [original_val] if original_val is not None else None

        self.raw_predicate =  [raw_predicate] if raw_predicate is not None else None
        self.raw_val = [raw_val] if raw_val is not None else None

if __name__ == '__main__':
    for tag in tags:
        rng = np.random.RandomState(seed)
        os.makedirs('./MSCN/', exist_ok=True)
        tables_name = [name.replace('.csv', '') for name in JoinOrderBenchmark.GetJobLightJoinKeys().keys()]
        tables_dict: Dict[str, common.CsvTable] = MakeIMDBTables(tables_name, data_dir='../datasets/job/', use_cols=use_cols,tag=tag)
        print(tables_name)
        # tables_dict = MakeIMDBTables(tables_name, data_dir='../datasets/job/', use_cols='full')
        tables = [tables_dict[name] for name in tables_name]  # keep order to give determined generation that can be reproduced
        global_distinct_values = set()
        for dataset in tables_name:
            table = tables_dict[dataset]
            key_col = table.columns_dict[JoinOrderBenchmark.GetJobLightJoinKeys()[dataset]]
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
            key_col = table.columns_dict[JoinOrderBenchmark.GetJobLightJoinKeys()[dataset]]
            key_col.SetGlobalDiscretizeMask(global_distinct_values)
        # global_distinct_values = torch.as_tensor(global_distinct_values, device=util.get_device())
        estimator = DirectEstimator(None,
                                    tables_dict,
                                    tables_dict,
                                    global_distinct_values=global_distinct_values,
                                    config={'test': {'faster_version': True}},
                                    device=util.get_device())

        # generating samples for bitmaps
        for table in tables:
            sample_path = f'./MSCN/{table.name}_sample{sample_num}_{use_cols}{tag}.pkl'
            if not os.path.exists(sample_path):
                table.sample = table.data.sample(sample_num, replace=False)
                with open(sample_path, 'wb') as f:
                    pkl.dump(table.sample, f)
            else:
                print(f'loading samples from {sample_path}')
                with open(sample_path, 'rb') as f:
                    table.sample = pkl.load(f)


        join_queries = []
        raw_join_queries = []
        join_bitmaps = []
        i = 0
        true_cards = []
        join_how = []
        pbar = tqdm(total=workload_num, desc='generating queries')
        while i < workload_num:
            # how = rng.choice(hows)
            how = rng.choice(hows, 1, p=prob_hows).item() # '=' is more likely to get 0 card, by generating 10k queries, we only got 788 '=' queries, it's about 0.08. We expect 0.2, 2.5x higher, so the prob is 2.5x0.2=0.5.
            join_num = rng.randint(2, 6)
            join_tables_idx = rng.choice(np.arange(len(tables)), size=join_num, replace=False)
            join_tables = [tables[join_table].name for join_table in join_tables_idx]
            raw_join_query = [[tables[tab_idx].name for tab_idx in join_tables_idx], [], [], []]
            predicates = []
            used_tables = set()
            bitmaps = [np.ones(sample_num, dtype=bool) for _ in range(len(join_tables_idx))] # in order of join_tables
            for j, table_idx in enumerate(join_tables_idx):
                table = tables[table_idx]
                pred_num = rng.randint(0, len(table.columns))
                raw_query = gen_query_single_table(table.name, table, rng, how)
                if len(raw_query[0])==0:
                    continue
                raw_join_query[1].append(raw_query[0])
                raw_join_query[2].append(raw_query[1])
                raw_join_query[3].append(raw_query[2])
                # get sample's bitmap
                bitmap = bitmaps[j]
                for col_idx, (col_name, op, v) in enumerate(zip(raw_query[0], raw_query[1], raw_query[2])):
                    if op is not None:
                        bitmap &= AQP_estimator.OPS[op](table.sample[col_name], v)
                bitmaps[j] = bitmap.values if isinstance(bitmap, pd.DataFrame) or isinstance(bitmap, pd.Series) else bitmap
                for col_idx, (col_name, ops, vs) in enumerate(zip(raw_query[0], raw_query[1], raw_query[2])):
                    if ops is not None:
                        # col_idx = table.base_table.columns_name_to_idx[col_name]
                        original_projected_ops = AQP_estimator.OPS_dict[ops]
                        col = table.base_table.columns_dict[col_name]
                        original_projected_vals = col.ValToBin(vs)
                        predicates.append(Predicate(table.name, col_name,
                                                    original_projected_ops,
                                                    original_projected_vals,
                                                    ops,
                                                    vs))
            for table_name in join_tables:
                predicates.append(AQP_estimator.DirectEstimator.GetKeyPredicate(table_name, AQP_estimator.DirectEstimator.GetJoinKeyColumn(tables_dict[table_name])))
            # get true card
            condition_prob = estimator.get_prob_of_predicate_tree(predicates, join_tables, tables_dict, how, real=True)
            pred_card = int(condition_prob)
            if pred_card > 0:
                true_cards.append(pred_card)
                i += 1
                raw_join_queries.append(raw_join_query)
                join_how.append(how)
                join_bitmaps.append(np.vstack(bitmaps))
                pbar.update(1)
        pbar.close()
        with open(f'./MSCN/job-{workload_num}queries-seed{seed}-bitmaps{tag}.pkl', 'wb') as f:
            pkl.dump(join_bitmaps, f)
        final_queries_csv = ''
        for i, raw_query in tqdm(enumerate(raw_join_queries), desc='re-formating queries'):
            join_tables_idx = raw_query[0]
            raw_cols = raw_query[1]
            raw_ops = raw_query[2]
            raw_vals = raw_query[3]
            # table
            for table in join_tables_idx:
                final_queries_csv += f'{table} {shorten_table_name[table]},'
            final_queries_csv = final_queries_csv[:-1] + '#'
            # join key
            if len(join_tables_idx) == 1:
                final_queries_csv += '#'
            else:
                for table1, table2 in zip(join_tables_idx[:-1], join_tables_idx[1:]):
                    final_queries_csv += f'{shorten_table_name[table1]}.{JoinOrderBenchmark.GetJobLightJoinKeys()[table1]}{join_how[i]}{shorten_table_name[table2]}.{JoinOrderBenchmark.GetJobLightJoinKeys()[table2]},'
                final_queries_csv = final_queries_csv[:-1] + '#'
            # preds
            for j, (table, cols, join_ops, join_vals) in enumerate(zip(join_tables_idx, raw_cols, raw_ops, raw_vals)):
                for col, op, val in zip(cols, join_ops, join_vals):
                    if op is not None:
                        final_queries_csv += f"{shorten_table_name[join_tables_idx[j]]}.{col},{op},{tables_dict[table].columns_dict[col].ValToBin(val)},"
            final_queries_csv = final_queries_csv[:-1] + '#'
            # card
            final_queries_csv += f'{true_cards[i]}\n'
            if i == len(join_queries) - 1:
                final_queries_csv = final_queries_csv[:-1]
            with open(f'./MSCN/job-{workload_num}queries-seed{seed}{tag}.csv', 'w') as f:
                f.write(final_queries_csv)
