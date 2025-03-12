#!/usr/bin/env python3
"""Unbiased join sampler using the Exact Weight algorithm."""

import argparse
import collections
import os
import pickle
import time

import glog as log
import numpy as np
import pandas as pd

import common
import datasets
# import experiments
import factorized_sampler_lib.data_utils as data_utils

# Assuming the join columns contain only non-negative values.
NULL = -1

# ----------------------------------------------------------------
#      Column names utils
# ----------------------------------------------------------------


def get_jct_count_columns(join_spec):
    return get_fanout_columns_impl(join_spec, "{table}.{key}.cnt",
                                   "{table}.{key}.cnt")


def get_fanout_columns(join_spec):
    return get_fanout_columns_impl(join_spec, "__fanout_{table}",
                                   "__fanout_{table}__{key}")


def get_fanout_columns_impl(join_spec, single_key_fmt, multi_key_fmt):
    ret = []
    for t in join_spec.join_tables_idx:
        # todo support when join root needs to be downscaled.  E.g., Say the
        # tree looks like root -> A -> B. Now a query joins A join B. We still
        # want to downscale fanout introduced by root.
        if t == join_spec.join_root:
            continue
        keys = join_spec.join_keys[t]
        if len(keys) == 1:
            ret.append(single_key_fmt.format(table=t, key=keys[0]))
        else:
            for k in keys:
                ret.append(multi_key_fmt.format(table=t, key=k))
    return ret


# ----------------------------------------------------------------
#      Sampling from join count tables
# ----------------------------------------------------------------


def get_distribution(series):
    """Make a probability distribution out of a series of counts."""
    arr = series.values
    total = np.sum(arr)
    assert total > 0
    return arr / total


# ----------------------------------------------------------------
#      Sampling from data tables
# ----------------------------------------------------------------


def load_data_table(table, join_keys, usecols):
    return data_utils.load_table(table,
                                 usecols=usecols,
                                 dtype={k: np.int64 for k in join_keys})


# ----------------------------------------------------------------
#      Main Sampler
# ----------------------------------------------------------------


def load_jct(table, join_name):
    return data_utils.load(f"{join_name}/{table}.jct",
                           f"join count table of `{table}`")


def _make_sampling_table_ordering(tables, root_name):
    """
    Returns a list of table names with the join_root at the front.
    """
    return [root_name
           ] + [table.name for table in tables if table.name != root_name]




LoadedTable = collections.namedtuple("LoadedTable", ["name", "data"])


# def main():
#     config = experiments.JOB_FULL
#     join_spec = join_utils.get_join_spec(config)
#     prepare_utils.prepare(join_spec)
#     loaded_tables = []
#     for t in join_spec.join_tables:
#         print('Loading', t)
#         table = datasets.LoadImdb(t, use_cols=config["use_cols"])
#         table.data.info()
#         loaded_tables.append(table)
#
#     t_start = time.time()
#     join_iter_dataset = FactorizedSamplerIterDataset(
#         loaded_tables,
#         join_spec,
#         sample_batch_size=1000 * 100,
#         disambiguate_column_names=True)
#
#     table = common.ConcatTables(loaded_tables,
#                                 join_spec.join_keys,
#                                 sample_from_join_dataset=join_iter_dataset)
#
#     join_iter_dataset = common.FactorizedSampleFromJoinIterDataset(
#         join_iter_dataset,
#         base_table=table,
#         factorize_blacklist=[],
#         word_size_bits=10,
#         factorize_fanouts=True)
#     t_end = time.time()
#     log.info(f"> Initialization took {t_end - t_start} seconds.")
#
#     join_iter_dataset.join_iter_dataset._sample_batch()
#     print('-' * 60)
#     print("Done")
#
#
# if __name__ == "__main__":
#     main()
