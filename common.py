"""Data abstractions."""
import copy
import time
from collections import defaultdict
from typing import List, Any
import itertools
import numpy as np
import pandas as pd

import torch
from torch.utils import data
from torch.utils.data import WeightedRandomSampler, IterableDataset

import datasets
from utils import util
import mysamplerAQP as mysampler

range_fn = [None, torch.less, torch.greater, torch.less_equal, torch.greater_equal]

# Na/NaN/NaT Semantics
#
# Some input columns may naturally contain missing values.  These are handled
# by the corresponding numpy/pandas semantics.
#
# Specifically, for any value (e.g., float, int, or np.nan) v:
#
#   np.nan <op> v == False.
#
# This means that in progressive sampling, if a column's domain contains np.nan
# (at the first position in the domain), it will never be a valid sample
# target.
#
# The above evaluation is consistent with SQL semantics.
from utils.util import get_device

TYPE_NORMAL_ATTR = 0
TYPE_INDICATOR = 1
TYPE_FANOUT = 2
PREDICATE_OPS = [
    '=',
    '>',
    '<',
    '>=',
    "<="
]
PROJECT_OPERATORS = {
    "<": "<",
    ">": ">",
    "!=": "ALL_TRUE",
    "<=": "<=",
    ">=": ">=",
}
# What each operator projects to for the last subvar, if not the same as other
# subvars.
PROJECT_OPERATORS_LAST = {
    "<": "<",
    ">": ">",
    "!=": "!=",
}
# What the dominant operator for each operator is.
PROJECT_OPERATORS_DOMINANT = {
    "<=": "<",
    ">=": ">",
    "<": "<",
    ">": ">",
    "!=": "!=",
}

IMDB_file_col_name = {
    'aka_name': ['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
    'aka_title': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code',
                  'episode_of_id', 'season_nr', 'episode_nr', 'note', 'md5sum'],
    'cast_info': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
    'char_name': ['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
    'comp_cast_type': ['id', 'kind'],
    'company_name': ['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
    'company_type': ['id', 'kind'],
    'complete_cast': ['id', 'movie_id', 'subject_id', 'status_id'],
    'info_type': ['id', 'info'],
    'keyword': ['id', 'keyword', 'phonetic_code'],
    'kind_type': ['id', 'kind'],
    'link_type': ['id', 'link'],
    'movie_companies': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
    'movie_info_idx': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
    'movie_keyword': ['id', 'movie_id', 'keyword_id'],
    'movie_link': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
    'name': ['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode',
             'md5sum'],
    'role_type': ['id', 'role'],
    'title': ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id',
              'season_nr', 'episode_nr', 'series_years', 'md5sum'],
    'movie_info': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
    'person_info': ['id', 'person_id', 'info_type_id', 'info', 'note']}

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

class Column(object):
    """A column.  Data is write-once, immutable-after.

    Typical usage:
      col = Column('myCol').Fill(data).SetDistribution(domain_vals)

    "data" and "domain_vals" are NOT copied.
    """

    def __init__(self,
                 name,
                 raw_col_name=None,
                 distribution_size=None,
                 pg_name=None,
                 factor_id=None,
                 is_last_fact_col=True,
                 bit_width=None,
                 bit_offset=0,
                 domain_bits=None,
                 num_bits=None):
        self.name = name
        self.raw_col_name = raw_col_name if raw_col_name is not None else name

        # Data related fields.
        self.data = None
        self.all_distinct_values = None
        self.all_distinct_values_gpu = None
        self.all_discretize_distinct_values = None
        self.all_discretize_distinct_values_gpu = None
        self.global_discretize_distinct_values_mask = None
        self.distribution_size = distribution_size
        # self.freq = None
        # self.dist_weights = None
        self.has_none = False
        self.nan_ind = None

        # Factorization related fields.
        self.factor_id = factor_id
        self.is_last_fact_col = is_last_fact_col
        self.bit_width = bit_width if bit_width is not None else len(bin(self.distribution_size))-2 if self.distribution_size is not None else None
        self.bit_offset = bit_offset
        self.domain_bits = domain_bits
        self.num_bits = num_bits
        # self.disc_dvs2idx = {}

        # pg_name is the name of the corresponding column in the Postgres db.
        if pg_name:
            self.pg_name = pg_name
        else:
            self.pg_name = name

    def SetGlobalDiscretizeMask(self, global_discretize_distinct_values: np.ndarray):
        self.global_discretize_distinct_values_mask = torch.as_tensor(np.isin(global_discretize_distinct_values, self.all_distinct_values), device=util.get_device())

    def Name(self):
        """Name of this column."""
        return self.name

    def DistributionSize(self):
        """This column will take on discrete values in [0, N).

        Used to dictionary-encode values to this discretized range.
        """
        return self.distribution_size

    def ProjectValue(self, value):
        """Bit slicing: returns the relevant bits in binary for a sub-var."""
        assert self.factor_id is not None, "Only for factorized cols"
        return (value >> self.bit_offset) & (2 ** self.bit_width - 1)

    def ProjectOperator(self, op):
        assert self.factor_id is not None, "Only for factorized cols"
        if self.bit_offset > 0:
            # If not found, no need to project.
            return PROJECT_OPERATORS.get(op, op)
        # Last subvar: identity (should not project).
        return op

    def ProjectOperatorDominant(self, op):
        assert self.factor_id is not None, "Only for factorized cols"
        return PROJECT_OPERATORS_DOMINANT.get(op, op)

    def BinToVal(self, bin_id):
        if bin_id is None:
            return None
        assert bin_id >= 0 and bin_id < self.distribution_size, bin_id
        return self.all_distinct_values[bin_id]

    def ValToBin(self, val):
        if val is None:
            return val
        # 要求all_distince_values已排序
        # try:
        if not self.has_none:
            res = np.searchsorted(self.all_distinct_values, val)
        else:
            if val is not np.nan:
                res = np.searchsorted(self.all_distinct_values[1:], val) + 1
            else:
                res = 0
        # except Exception as e:
            # print(e)
            # print(f'val: {val}, col:{self}, dv: {self.all_distinct_values}')
        return res

    def FindProjection(self, val):
        if val in self.all_distinct_values:
            return (self.ValToBin(val), True)
        elif val > self.all_distinct_values[-1]:
            return (len(self.all_distinct_values), False)
        elif val < self.all_distinct_values[0]:
            return (-1, False)
        else:
            return (next(
                i for i, v in enumerate(self.all_distinct_values) if v > val),
                    False)

    def SetDistribution(self, distinct_values, freq=None):
        """This is all the values this column will ever see."""
        assert self.all_distinct_values is None
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(distinct_values)
        self.nan_ind = is_nan
        contains_nan = np.any(is_nan)
        self.has_none = contains_nan
        dv_no_nan = distinct_values[~is_nan]
        if freq is not None:
            freq_no_nan = freq[~is_nan]
            freq_nan = freq[is_nan]
        # IMPORTANT: np.sort puts NaT values at beginning, and NaN values
        # at end for our purposes we always add any null value to the
        # beginning.
        unq_dv_no_nan = np.unique(dv_no_nan)
        idx = np.argsort(unq_dv_no_nan)
        vs = unq_dv_no_nan[idx]
        if freq is not None:
            freq_no_nan = freq_no_nan[idx]
        # assert (vs == np.sort(np.unique(dv_no_nan))).all()
        if contains_nan:
            if np.issubdtype(distinct_values.dtype, np.datetime64):
                vs = np.insert(vs, 0, np.datetime64('NaT'))
            else:
                vs = np.insert(vs, 0, np.nan)
            if freq is not None:
                freq = np.insert(freq_no_nan, 0, freq_nan.item())
        # if freq is not None:
        #     self.freq = torch.as_tensor(freq, dtype=torch.float32)
        #     self.dist_weights = (1 / self.freq)
        #     self.dist_weights /= self.dist_weights.mean()
        #     self.dist_weights.requires_grad = False
        # if self.distribution_size is not None:
        #     assert len(vs) == self.distribution_size
        self.all_distinct_values = vs
        if distinct_values.dtype != np.object_ and distinct_values.dtype != np.datetime64 and not np.issubdtype(
                distinct_values.dtype, np.datetime64):
            self.all_distinct_values_gpu = torch.as_tensor(vs, device=get_device())
        else:
            self.all_distinct_values_gpu = vs
        self.distribution_size = len(vs)
        # [前提]：pd.Categorical保序且Discritze时dvs有序，nan在最前
        self.all_discretize_distinct_values = np.arange(self.distribution_size)
        self.all_discretize_distinct_values_gpu = torch.as_tensor(self.all_discretize_distinct_values,
                                                                  device=get_device())
        if self.bit_width is None:
            self.bit_width = len(bin(self.distribution_size-1)) - 2
        return self

    def Fill(self, data_instance, infer_dist=False):
        assert self.data is None
        self.data = data_instance
        # If no distribution is currently specified, then infer distinct values
        # from data.
        if infer_dist:
            self.SetDistribution(self.data)
        return self

    def InsertNullInDomain(self):
        # Convention: np.nan would only appear first.
        if not pd.isnull(self.all_distinct_values[0]):
            if self.all_distinct_values.dtype == np.dtype('object'):
                # String columns: inserting nan preserves the dtype.
                self.all_distinct_values = np.insert(self.all_distinct_values,
                                                     0, np.nan)
            else:
                # Assumed to be numeric columns.  np.nan is treated as a
                # float.
                self.all_distinct_values = np.insert(
                    self.all_distinct_values.astype(np.float64, copy=False), 0,
                    np.nan)
            self.distribution_size = len(self.all_distinct_values)

    def __repr__(self):
        return 'Column({}, distribution_size={})'.format(
            self.name, self.distribution_size)


class Table(object):
    """A collection of Columns."""

    def __init__(self, name, columns, pg_name=None, validate_cardinality=True):
        """Creates a Table.

        Args:
            name: Name of this table object.
            columns: List of Column instances to populate this table.
            pg_name: name of the corresponding table in Postgres.
        """
        self.name = name
        if validate_cardinality:
            self.cardinality = self._validate_cardinality(columns)
        else:
            # Used as a wrapper, not a real table.
            self.cardinality = None
        self.columns = columns

        # Bin to val funcs useful for sampling.  Takes
        #   (col 1's bin id, ..., col N's bin id)
        # and converts it to
        #   (col 1's val, ..., col N's val).
        self.column_bin_to_val_funcs = [c.BinToVal for c in columns]
        self.val_to_bin_funcs = [c.ValToBin for c in columns]

        self.name_to_index = {c.Name(): i for i, c in enumerate(self.columns)}
        self.columns_size = np.array([col.distribution_size for col in columns])

        if pg_name:
            self.pg_name = pg_name
        else:
            self.pg_name = name

    def __repr__(self):
        return '{}({})'.format(self.name, self.columns)

    def _validate_cardinality(self, columns):
        """Checks that all the columns have same the number of rows."""
        cards = [len(c.data) for c in columns]
        c = np.unique(cards)
        assert len(c) == 1, c
        return c[0]

    def to_df(self):
        return pd.DataFrame({c.name: c.data for c in self.columns})

    def Name(self):
        """Name of this table."""
        return self.name

    def Columns(self):
        """Return the list of Columns under this table."""
        return self.columns

    def ColumnIndex(self, name):
        """Returns index of column with the specified name."""
        assert name in self.name_to_index, (name,
                                            list(self.name_to_index.keys()))
        return self.name_to_index[name]

    def __getitem__(self, column_name):
        return self.columns[self.name_to_index[column_name]]

    def TableColumnIndex(self, source_table, col):
        """Returns index of column with the specified name/source."""
        name = JoinTableAndColumnNames(source_table, col)
        assert name in self.name_to_index, (name,
                                            list(self.name_to_index.keys()))
        return self.name_to_index[name]


class CsvTable(Table):
    """Wraps a CSV file or pd.DataFrame as a Table."""

    def __init__(self,
                 name,
                 filename_or_df,
                 cols,
                 type_casts={},
                 pg_name=None,
                 pg_cols=None,
                 **kwargs):
        """Accepts the same arguments as pd.read_csv().

        Args:
            filename_or_df: pass in str to reload; otherwise accepts a loaded
              pd.Dataframe.
            cols: list of column names to load; can be a subset of all columns.
            type_casts: optional, dict mapping column name to the desired numpy
              datatype.
            pg_name: optional str, a convenient field for specifying what name
              this table holds in a Postgres database.
            pg_name: optional list of str, a convenient field for specifying
              what names this table's columns hold in a Postgres database.
            **kwargs: keyword arguments that will be pass to pd.read_csv().
        """
        self.name = name
        self.pg_name = pg_name
        self.base_table = self
        if isinstance(filename_or_df, str):
            self.data = self._load(filename_or_df, cols, **kwargs, low_memory=False)
        else:
            assert (isinstance(filename_or_df, pd.DataFrame))
            self.data = filename_or_df
        self.bounded_distinct_value = None
        self.bounded_col = None
        self.columns = self._build_columns(self.data, cols, type_casts, pg_cols)
        self.columns_dict = {}
        self.columns_name_to_idx = {}
        self.map_from_col_to_fact_col = {}
        self.map_from_fact_col_to_col = {}
        for idx, col in enumerate(self.columns):
            self.columns_dict[col.name] = col
            self.map_from_col_to_fact_col[col.name] = [col.name]
            self.map_from_fact_col_to_col[col.name] = col.name
            self.columns_name_to_idx[col.name] = idx
        super(CsvTable, self).__init__(name, self.columns, pg_name)

    def getConditionalDistribution(self, col_name, conditions: dict):
        key = list(conditions.keys())[0]
        mask = self.data[key] == conditions[key]
        for key in conditions.keys():
            mask = mask & (self.data[key] == conditions[key])
        picked = self.data[mask][col_name]  # 满足条件的
        return picked.GetDistribution()

    def _load(self, filename, cols, **kwargs):
        print('Loading csv...', end=' ')
        s = time.time()
        df = pd.read_csv(filename, usecols=cols, **kwargs)
        if cols is not None:
            df = df[cols]
        print('done, took {:.1f}s'.format(time.time() - s))
        return df

    def _build_columns(self, data, cols, type_casts, pg_cols):
        """Example args:

            cols = ['Model Year', 'Reg Valid Date', 'Reg Expiration Date']
            type_casts = {'Model Year': int}

        Returns: a list of Columns.
        """
        print('Parsing...', end=' ')
        s = time.time()
        for col, typ in type_casts.items():
            if col not in data:
                continue
            if typ != np.datetime64:
                data[col] = data[col].astype(typ, copy=False)
            else:
                # Both infer_datetime_format and cache are critical for perf.
                data[col] = pd.to_datetime(data[col],
                                           infer_datetime_format=True,
                                           cache=True)

        # Discretize & create Columns.
        if cols is None:
            cols = data.columns
        columns = []
        if pg_cols is None:
            pg_cols = [None] * len(cols)
        for c, p in zip(cols, pg_cols):
            col = Column(c, pg_name=p)
            col.Fill(data[c])

            # dropna=False so that if NA/NaN is present in data,
            # all_distinct_values will capture it.
            #
            # For numeric: np.nan
            # For datetime: np.datetime64('NaT')
            tmp = data[c].value_counts(dropna=False)
            col.SetDistribution(tmp.index.values, tmp.values / len(data[c]))
            columns.append(col)
        print('done, took {:.1f}s'.format(time.time() - s))
        return columns


class CacheDataset(data.Dataset):
    def __init__(self, datas: List[List[Any]]):
        super().__init__()
        self.length = len(datas[0])
        # 检查所有数据列表长度一致
        # for d in datas[1:]:
        #     assert len(d) == self.length, "All data lists must have the same length"
        # 预重组数据：将多个列表按索引打包成样本元组
        # 例如 datas = [[x1, x2, x3], [y1, y2, y3]] -> samples = [(x1, y1), (x2, y2), (x3, y3)]
        self.samples = list(zip(*datas))
        self.idx = 0
        self.run_out = False

    def __len__(self):
        return self.length

    def __getitem__(self, _: int) -> tuple:
        # 直接通过索引返回预打包的样本，时间复杂度O(1)
        res = self.samples[self.idx]
        self.idx += 1
        if self.idx>=self.length:
            self.run_out = True
            self.idx = 0
        return res

class TableDataset(data.Dataset):
    """Wraps a Table and yields each row as a PyTorch Dataset element."""

    def __init__(self, table: CsvTable, bs, expand_factor=1, queries=None, true_cards=None, inp=None,
                 valid_i_list=None, wild_card_mask=None, model=None):
        super(TableDataset, self).__init__()
        self.base_table = table #copy.deepcopy(table)
        self.columns = self.base_table.columns
        self.expand_factor = expand_factor
        self.bs = bs
        self.num_queries = len(true_cards) if true_cards is not None else 1
        if queries is not None:
            assert len(queries[0]) == len(true_cards)
        self.queries = [np.array(queries[1]), np.array(queries[2])] if queries else None
        self.wild_card_mask = wild_card_mask
        self.true_cards = np.array(true_cards)
        self.model = model
        self.inp = inp
        self.valid_i_list = valid_i_list
        print('Discretizing table...', end=' ')
        s = time.time()
        # [cardianlity, num cols].
        self.tuples_np = np.stack(
            [self.Discretize(c) for c in self.base_table.Columns()], axis=1)
        self.tuples = torch.as_tensor(self.tuples_np.astype(np.float32, copy=False)).long().pin_memory()

        self.device = util.get_device()
        self.new_tuples = torch.tile(
            torch.zeros((self.bs * self.expand_factor, self.tuples.shape[-1]), requires_grad=False,
                        dtype=torch.long).unsqueeze(1), dims=(1, 2, 1)).pin_memory() - 1
        self.preds = torch.zeros((self.bs * self.expand_factor, 2, 5 * len(self.base_table.columns)), requires_grad=False,
                                 dtype=torch.long).pin_memory()
        self.has_nones = [c.has_none for c in self.base_table.columns]
        self.columns_size = [c.distribution_size for c in self.base_table.columns]
        self.map_from_col_to_fact_col = self.base_table.map_from_col_to_fact_col
        self.columns_name_to_idx = self.base_table.columns_name_to_idx
        print('done, took {:.1f}s'.format(time.time() - s))

    def Discretize(self, col):
        """Discretize values into its Column's bins.

        Args:
          col: the Column.
        Returns:
          col_data: discretized version; an np.ndarray of type np.int32.
        """
        return Discretize(col)

    def size(self):
        return len(self.tuples)

    def __len__(self):
        if self.queries is not None:
            return max(len(self.tuples), len(self.queries))
        else:
            return len(self.tuples)

    def __getitem__(self, idx):
        return idx

    def collect_fn(self, samples):
        from model import made
        with torch.no_grad():
            num_samples = len(samples)
            tuples_idxs = torch.as_tensor(samples, dtype=torch.long)
            if self.queries:
                # queries_idx = np.random.choice(range(len(self.queries[0])), num_samples, replace=False)
                # queries = [self.queries[0][queries_idx], self.queries[1][queries_idx]]
                # true_cards = self.true_cards[queries_idx]
                # wild_card_mask = self.wild_card_mask[queries_idx].to(self.device, non_blocking=True)
                # inps = self.inp[queries_idx].to(self.device, non_blocking=True)
                query_start = np.random.randint(len(self.queries[0]))
                query_end = query_start + num_samples
                if query_end > len(self.queries[0]):
                    query_end = len(self.queries[0])
                queries = [self.queries[0][query_start:query_end], self.queries[1][query_start:query_end]]
                true_cards = self.true_cards[query_start:query_end]
                wild_card_mask = self.wild_card_mask[query_start:query_end].to(self.device, non_blocking=True)
                inps = self.inp[query_start:query_end].to(self.device, non_blocking=True)
                # 更新inp中的wild_card
                for i in range(self.model.nin):
                    if i == 0:
                        s = 0
                    else:
                        s = self.model.input_bins_encoded_cumsum[i - 1]
                    e = self.model.input_bins_encoded_cumsum[i]
                    # torch与numpy不同，mask与slice合用时mask只能为一维向量
                    if isinstance(self.model, made.MADE):
                        pred_shift = 5
                    else:
                        pred_shift = 0
                    inps[wild_card_mask[:, 0, i], 0, s:e - pred_shift] = 0
                    inps[wild_card_mask[:, 1, i], 1, s:e - pred_shift] = 0
                    inps[..., s:e - pred_shift] = inps[..., s:e - pred_shift] + wild_card_mask.narrow(-1, i, 1).float() * self.model.unk_embeddings[i]
                valid_i_lists = [self.valid_i_list[i][query_start:query_end] for i in range(len(self.valid_i_list))]
                # valid_i_lists = [self.valid_i_list[i][queries_idx] for i in range(len(self.valid_i_list))]

            tuples = self.tuples[tuples_idxs]
            if self.expand_factor > 1:
                tuples = torch.tile(tuples, dims=(self.expand_factor, 1))
            num_samples = tuples.shape[0]
            # -1 by default
            new_tuples = self.new_tuples[:num_samples]
            new_tuples.fill_(-1)
            new_preds = self.preds[:num_samples]
            new_preds.zero_()
            mysampler.sample(tuples, new_tuples, new_preds, self.columns_size, self.has_nones, num_samples, multi_preds=True, bounded_eqv_col_idx=-1)
            tuples = tuples.to(self.device, non_blocking=True)
            new_tuples = new_tuples.to(self.device, non_blocking=True)
            new_preds = new_preds.to(self.device, non_blocking=True)
            # for i, column_size in enumerate(self.columns_size):
            #     if new_tuples[...,i].max().item()>=column_size:
            #         print(torch.where(new_tuples[...,i]>=column_size))
            #         assert False
            # util.check_sample(self.table, tuples, new_tuples, new_preds)
            if self.queries is None:
                return [new_tuples, new_preds, tuples]
            else:
                return [new_tuples, new_preds, tuples, queries, true_cards, inps, valid_i_lists]


class FactorizedTable(data.Dataset):
    """Wraps a TableDataset to factorize large-card columns."""

    def __init__(self, table_dataset, word_size_bits=5, factorize_blacklist=[]):
        assert isinstance(table_dataset, TableDataset), table_dataset
        self.table_dataset = table_dataset
        self.device = util.get_device()
        self.base_table = self.table_dataset.base_table
        self.word_size_bits = word_size_bits
        self.word_size = 2 ** self.word_size_bits
        self.map_from_col_to_fact_col = {}
        self.map_from_fact_col_to_col = None
        self.ori_id_to_fact_id = defaultdict(list)
        self.fact_id_to_ori_id = {}
        self.fact_col_number = []
        self.fact_col_mapping = defaultdict(list)
        self.factorize_blacklist = factorize_blacklist
        self.columns, self.factorized_tuples_np = self._factorize(
            self.table_dataset.tuples_np)
        self.columns_dict: dict[str, Column] = {}
        self.columns_name_to_idx = {}
        self.columns_size = [c.distribution_size for c in self.base_table.columns]
        for idx, col in enumerate(self.columns):
            self.columns_dict[col.name] = col
            self.columns_name_to_idx[col.name] = idx
        self.factorized_tuples = torch.as_tensor(
            self.factorized_tuples_np.astype(copy=False, dtype=int))
        self.has_nones=[]
        for col in self.table_dataset.columns:
            # for ii in range(len(self.map_from_col_to_fact_col[col.name])):
                self.has_nones.append(col.has_none)
        self.cardinality = table_dataset.base_table.cardinality
        self.new_tuples = torch.tile(
            torch.zeros((self.table_dataset.bs * self.table_dataset.expand_factor, self.table_dataset.tuples.shape[-1]), requires_grad=False,
                        dtype=torch.long).unsqueeze(1), dims=(1, 2, 1)).pin_memory() - 1
        self.preds = torch.zeros((self.table_dataset.bs * self.table_dataset.expand_factor, 2, 5 * self.table_dataset.tuples.shape[-1]), requires_grad=False,
                                 dtype=torch.long).pin_memory()
        self.num_facts_per_col = None

        self.bit_widths = torch.tensor([fact_col.bit_width for fact_col in self.columns], dtype=torch.int64, device=self.device)
        self.bit_offsets = torch.tensor([fact_col.bit_offset for fact_col in self.columns], dtype=torch.int64, device=self.device)
        self.bit_masks = (1 << self.bit_widths) - 1
        self.last_facts_idx = [0] * len(self.base_table.columns)
        for c, fact_cs in self.map_from_col_to_fact_col.items():
            c_id = self.base_table.columns_name_to_idx[c]
            self.last_facts_idx[c_id] = self.columns_name_to_idx[fact_cs[-1]]
            for fact_c in fact_cs:
                fact_id = self.columns_name_to_idx[fact_c]
                self.ori_id_to_fact_id[c_id].append(fact_id)
                self.fact_id_to_ori_id[fact_id] = c_id
        key_col_name = datasets.JoinOrderBenchmark.GetJobLightJoinKeys()[self.base_table.name]
        self.key_col_ori_idx = self.base_table.columns_name_to_idx[key_col_name]
        # self.new_tuples = torch.tile(
        #     torch.zeros((self.bs * self.expand_factor, self.factorized_tuples.shape[-1]), requires_grad=False,
        #                 dtype=torch.long).unsqueeze(1), dims=(1, 2, 1)).pin_memory() - 1
        # self.preds = torch.zeros((self.bs * self.expand_factor, 2, 5 * len(self.columns)), requires_grad=False,
        #                          dtype=torch.long).pin_memory()

    def _factorize(self, tuples_np):
        """Factorize K columns into N>K columns based on word size."""
        factorized_data = []
        cols = []

        for i, col in enumerate(self.table_dataset.base_table.Columns()):
            dom = col.DistributionSize()
            #  or col.name not in ['id', 'movie_id']
            if dom <= self.word_size or col.name in self.factorize_blacklist:
                factorized_data.append(tuples_np[:, i])
                new_col = Column(col.name,
                                 distribution_size=col.distribution_size)
                new_col.SetDistribution(col.all_distinct_values)
                cols.append(new_col)
                self.map_from_col_to_fact_col[col.name] = [col.name]
                self.fact_col_number.append(1)
                print("col", i, col.name, "not factorized")
            else:
                if col.name in ['id', 'movie_id']:
                    word_size_bits = len(bin(dom)) // 2
                    word_size = 2 ** word_size_bits
                else:
                    word_size_bits = self.word_size_bits
                    word_size = self.word_size
                domain_bits = num_bits = len(bin(dom)) - 2
                # fact_num = int(np.floor(domain_bits / self.word_size_bits) + 1)
                # word_size_bits = int(np.floor(domain_bits / fact_num)+1)
                # word_size = 2 ** word_size_bits
                word_mask = word_size - 1
                j = 0
                while num_bits > 0:  # slice off the most significant bits
                    bit_width = min(num_bits, word_size_bits)
                    num_bits -= word_size_bits
                    if num_bits < 0:
                        factorized_data.append(tuples_np[:, i] &
                                               (word_mask >> -num_bits))
                        dist_size = len(np.unique(factorized_data[-1]))
                        assert dist_size <= 2 ** (word_size_bits + num_bits)
                        f_col = Column(col.name + "_fact_" + str(j),
                                       raw_col_name=col.name,
                                       distribution_size=dist_size,
                                       factor_id=j,
                                       is_last_fact_col=True,
                                       bit_width=bit_width,
                                       bit_offset=0,
                                       domain_bits=domain_bits,
                                       num_bits=num_bits)
                        if col.name in self.map_from_col_to_fact_col.keys():
                            self.map_from_col_to_fact_col[col.name].append(f_col.name)
                        else:
                            self.map_from_col_to_fact_col[col.name] = [f_col.name]
                    else:
                        # 这里先右移num_bits位，再与word_mask做与操作，即取出最高位的word_size_bits位
                        # num_bits位每次迭代减小，即每次右移更少位数，取更低位的
                        # 此处可看出fact分解顺序是从高位到低位
                        factorized_data.append((tuples_np[:, i] >> num_bits) &
                                               word_mask)
                        dist_size = len(np.unique(factorized_data[-1]))
                        assert dist_size <= word_size
                        f_col = Column(col.name + "_fact_" + str(j),
                                       raw_col_name=col.name,
                                       distribution_size=dist_size,
                                       factor_id=j,
                                       is_last_fact_col=True if num_bits==0 else False,
                                       bit_width=bit_width,
                                       bit_offset=num_bits,
                                       domain_bits=domain_bits,
                                       num_bits=num_bits)
                        if col.name in self.map_from_col_to_fact_col.keys():
                            self.map_from_col_to_fact_col[col.name].append(f_col.name)
                        else:
                            self.map_from_col_to_fact_col[col.name] = [f_col.name]
                    f_col.SetDistribution(factorized_data[-1])
                    self.fact_col_mapping[col].append(f_col)
                    cols.append(f_col)
                    print("fact col", i, num_bits, factorized_data[-1])
                    j += 1
                self.fact_col_number.append(len(self.fact_col_mapping[col]))
                print("orig", i, tuples_np[:, i])

        print("Factored table", cols)
        self.map_from_fact_col_to_col = dict((v_, k) for k, v in self.map_from_col_to_fact_col.items() for v_ in v)
        return cols, np.stack(factorized_data, axis=1)

    def size(self):
        return self.factorized_tuples_np.shape[0]

    def __len__(self):
        return self.factorized_tuples_np.shape[0]

    def __getitem__(self, idx):
        return idx

    def collect_fn(self, samples):
        key_col_idx = self.key_col_ori_idx
        if self.num_facts_per_col is None:
            self.num_facts_per_col = torch.tensor(self.fact_col_number, dtype=torch.long, device=self.device)
        with torch.no_grad():
            # 从原始表中采样谓词，返回采样后的原始表谓词和对应的factorized_tuples
            tuples_idxs = torch.as_tensor(samples, dtype=torch.long)
            tuples = self.table_dataset.tuples[tuples_idxs]
            factorized_tuples = self.factorized_tuples[tuples_idxs]
            if self.table_dataset.expand_factor > 1:
                tuples = torch.tile(tuples, dims=(self.table_dataset.expand_factor, 1))
                factorized_tuples = torch.tile(factorized_tuples, dims=(self.table_dataset.expand_factor, 1))
            num_samples = tuples.shape[0]
            # -1 by default
            new_tuples = self.new_tuples[:num_samples]
            new_tuples.fill_(-1)
            new_preds = self.preds[:num_samples]
            new_preds.zero_()

            # tuples = tuples.to(self.device, non_blocking=True)
            st = time.time()
            mysampler.sample(tuples, new_tuples, new_preds, self.columns_size, self.has_nones, num_samples, multi_preds=True, bounded_eqv_col_idx=key_col_idx)
            gen_cost = time.time() - st
            factorized_tuples = factorized_tuples.to(self.device, non_blocking=True)
            new_tuples = new_tuples.to(self.device, non_blocking=True)
            new_preds = new_preds.to(self.device, non_blocking=True)

            new_preds = new_preds.view(new_preds.shape[0], new_preds.shape[1], new_tuples.shape[-1], 5)

            # 生成随机mask，以训练wildcard_skip
            bs = tuples.shape[0]
            n_col = tuples.shape[1]
            for col_id in range(n_col):
                wildcard_mask = (1-torch.clamp(
                    torch.dropout(torch.ones(bs, device=self.device), p=1. - np.random.randint(
                        1, n_col + 1) * 1. / n_col, train=True)
                    , 0, 1)).bool()
                new_tuples[wildcard_mask,:, col_id] = -1
                new_preds[wildcard_mask, :, col_id,:] = 0

            # 这里防止key列生成多个谓词
            new_tuples[:,1, key_col_idx] = -1
            new_preds[:,1, key_col_idx*5:(key_col_idx+1)*5] = 0

            valid_pred_mask = new_tuples != -1

            # 扩展原始值到分解维度
            num_facts_per_col = self.num_facts_per_col
            raw_vals = new_tuples.repeat_interleave(num_facts_per_col, dim=-1)  # (bs, num_preds, n_fact_cols)

            # 分解计算 (使用位运算)
            shifted = raw_vals >> self.bit_offsets.view(1, 1, -1)  # (bs, num_preds, n_fact_cols)
            fact_vals = (shifted & self.bit_masks.view(1, 1, -1)).float()
            fact_vals[(~valid_pred_mask).repeat_interleave(num_facts_per_col, dim=-1)] = -1.0
            # 分解后的谓词与原始谓词保持一致
            fact_ops = new_preds.repeat_interleave(num_facts_per_col, dim=-2)
            fact_ops = fact_ops.view(new_preds.shape[0], new_preds.shape[1], -1)  # (bs, num_preds, n_fact_cols*5)

            return [fact_vals, fact_ops, factorized_tuples], gen_cost


def Discretize(col, data=None, dvs=None):
    """Transforms data values into integers using a Column's vocab.

    Args:
        col: the Column.
        data: list-like data to be discretized.  If None, defaults to col.data.

    Returns:
        col_data: discretized version; an np.ndarray of type np.int32.
    """
    # pd.Categorical() does not allow categories be passed in an array
    # containing np.nan.  It makes it a special case to return code -1
    # for NaN values.

    if data is None:
        data = col.data

    # pd.isnull returns true for both np.nan and np.datetime64('NaT').
    isnan = pd.isnull(col.all_distinct_values if dvs is None else dvs)
    if isnan.any():
        # We always add nan or nat to the beginning.
        assert isnan.sum() == 1, isnan
        assert isnan[0], isnan
        if dvs is None:
            dvs = col.all_distinct_values[1:]
        else:
            dvs = dvs[1:]
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data)

        # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
        # add 1 to everybody
        bin_ids = bin_ids + 1
    else:
        # This column has no nan or nat values.
        if dvs is None:
            dvs = col.all_distinct_values
        else:
            dvs = dvs
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data), (len(bin_ids), len(data))

    bin_ids = bin_ids.astype(np.int32, copy=False)
    assert (bin_ids >= 0).all(), (col, data, bin_ids)
    return bin_ids


def ConcatTables(tables,
                 join_keys,
                 disambiguate_column_names=False,
                 sample_from_join_dataset=None):
    """Makes a dummy Table to represent the schema of a join result."""
    cols_in_join = sample_from_join_dataset.columns_in_join()
    names = [t.name for t in tables]
    table = Table('-'.join(names), cols_in_join, validate_cardinality=False)
    table.table_names = names
    return table


def JoinTableAndColumnNames(table_name, column_name, sep=':'):
    return '{}{}{}'.format(table_name, sep, column_name)
