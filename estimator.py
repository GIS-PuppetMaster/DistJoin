import copy
import functools
import time
import numpy as np
import torch
from dataclasses import dataclass

import common
from datasets import JoinOrderBenchmark

from model.made import MADE

from common import Table
from model import made, transformer

import pandas as pd
from typing import *
from enum import Enum
from torch.cuda.amp import autocast
from train import DataParallelPassthrough
from utils import train_utils, util
import utils
import utils.util

real_table = []
OPS_array = np.array(['=', '>', '<', '>=', '<='])

OPS_dict = {'=': 0,
            '>': 1,
            '<': 2,
            '>=': 3,
            '<=': 4}
OPS = {
    '=': np.equal,
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
}

# OPS = {
#     '>': np.greater,
#     '<': np.less,
#     '>=': np.greater_equal,
#     '<=': np.less_equal,
#     '=': np.equal,
#     '!=': np.not_equal,
# }

op_types = {
    0: 'and',
    1: 'or'
}

inner_joins = {
    0: '=',
    1: '>',
    2: '<',
    3: '>=',
    4: '<='
}

torch_OPS = np.array([torch.eq, torch.greater, torch.less, torch.greater_equal, torch.less_equal])


def ConvertLikeToIn(fact_table, columns, operators, vals):
    """Pre-processes a query by converting LIKE predicates to IN predicates.

    Columns refers to the original columns of the table.
    """
    fact_columns = fact_table.columns
    fact_column_names = [fc.name for fc in fact_columns]
    assert len(columns) == len(operators) == len(vals)
    for i in range(len(columns)):
        col, op, val = columns[i], operators[i], vals[i]
        # We don't convert if this column isn't factorized.
        # If original column name isn't in the factorized col names,
        # then this column is factorized.
        # This seems sorta hacky though.
        if op is not None and col.name not in fact_column_names:
            assert len(op) == len(val)
            for j in range(len(op)):
                o, v = op[j], val[j]
                if 'LIKE' in o:
                    new_o = 'IN'
                    valid = OPS[o](col.all_distinct_values, v)
                    new_val = tuple(col.all_distinct_values[valid])
                    op[j] = new_o
                    val[j] = new_val

    assert len(columns) == len(operators) == len(vals)
    return columns, operators, vals


def ProjectQuery(fact_table, columns, operators, vals):
    """Projects these cols, ops, and vals to factorized version.

    Returns cols, ops, vals, dominant_ops where all lists are size
    len(fact_table.columns), in the factorized table's natural column order.

    Dominant_ops is None for all operators that don't use masking.
    <, >, <=, >=, IN all use masking.
    """
    columns, operators, vals = ConvertLikeToIn(fact_table, columns, operators,
                                               vals)
    nfact_cols = len(fact_table.columns)
    fact_cols = []
    fact_ops = []
    fact_vals = []
    fact_dominant_ops = []

    # i is the index of the current factorized column.
    # j is the index of the actual table column that col_i corresponds to.
    j = -1
    for i in range(nfact_cols):
        col = fact_table.columns[i]
        if col.factor_id in [None, 0]:
            j += 1
        op = operators[j]
        fact_cols.append(col)
        if op is None:
            fact_ops.append(None)
            fact_vals.append(None)
            fact_dominant_ops.append(None)
        else:
            val = vals[j]
            fact_ops.append([])
            fact_vals.append([])
            fact_dominant_ops.append([])
            for o, v in zip(op, val):
                if col.factor_id is None:
                    # This column is not factorized.
                    fact_ops[i].append(o)
                    fact_vals[i].append(v)
                    fact_dominant_ops[i] = None
                else:
                    # This column is factorized.  For IN queries, we need to
                    # map the projection overall elements in the tuple.
                    '''
                    if o == 'IN':
                        if len(v) == 0:
                            fact_ops[i].append('ALL_FALSE')
                            fact_vals[i].append(None)
                            fact_dominant_ops[i].append(None)
                        else:
                            fact_ops[i].append(o)
                            val_list = np.array(list(v))
                            val_list = common.Discretize(
                                columns[j], val_list, fail_out_of_domain=False)
                            assert len(val_list) > 0, val_list
                            p_v_list = np.vectorize(col.ProjectValue)(val_list)
                            fact_vals[i].append(tuple(p_v_list))
                            fact_dominant_ops[i].append('IN')
                    '''
                    # IS_NULL/IS_NOT_NULL Handling.
                    # IS_NULL+column has null value -> convert to = 0.
                    # IS_NULL+column has no null value -> return False for
                    #   everything.
                    # IS_NOT_NULL+column has null value -> convert to > 0.
                    # IS_NOT_NULL+column has no null value -> return True for
                    #   everything.
                    if 'NULL' in o:
                        if np.any(pd.isnull(columns[j].all_distinct_values)):
                            if o == 'IS_NULL':
                                fact_ops[i].append(col.ProjectOperator('='))
                                fact_vals[i].append(col.ProjectValue(0))
                                fact_dominant_ops[i].append(None)
                            elif o == 'IS_NOT_NULL':
                                fact_ops[i].append(col.ProjectOperator('>'))
                                fact_vals[i].append(col.ProjectValue(0))
                                fact_dominant_ops[i].append(
                                    col.ProjectOperatorDominant('>'))
                            else:
                                assert False, "Operator {} not supported".format(
                                    o)
                        else:
                            # No NULL values
                            if o == 'IS_NULL':
                                new_op = 'ALL_FALSE'
                            elif o == 'IS_NOT_NULL':
                                new_op = 'ALL_TRUE'
                            else:
                                assert False, "Operator {} not supported".format(
                                    o)
                            fact_ops[i].append(new_op)
                            fact_vals[i].append(None)  # This value is unused
                            fact_dominant_ops[i].append(None)
                    else:
                        # Handling =/<=/>=/</>.
                        # If the original column has a NaN, then we shoudn't
                        # include this in the result.  We can ensure this by
                        # adding a >0 predicate on the fact col.  Only need to
                        # do this if the original predicate is <, <=, or !=.
                        if o in ['<=', '<', '!='] and np.any(
                                pd.isnull(columns[j].all_distinct_values)):
                            fact_ops[i].append(col.ProjectOperator('>'))
                            fact_vals[i].append(col.ProjectValue(0))
                            fact_dominant_ops[i].append(
                                col.ProjectOperatorDominant('>'))
                        if v not in columns[j].all_distinct_values:
                            assert False, 'the design support it but this is not implemented in estimator yet'
                            # Handle cases where value is not in the column
                            # vocabulary.
                            assert o in ['=', '!=']
                            if o == '=':
                                # Everything should be False.
                                fact_ops[i].append('ALL_FALSE')
                                fact_vals[i].append(None)
                                fact_dominant_ops[i].append(None)
                            elif o == '!=':
                                # Everything should be True.
                                # Note that >0 has already been added,
                                # so there are no NULL results.
                                fact_ops[i].append('ALL_TRUE')
                                fact_vals[i].append(None)
                                fact_dominant_ops[i].append(None)
                        else:
                            # Distinct values of a factorized column is
                            # discretized.  So we do a lookup of the index for
                            # v in the original column, and not the fact col.
                            value = np.nonzero(
                                columns[j].all_distinct_values == v)[0][0]
                            p_v = col.ProjectValue(value)
                            p_op = col.ProjectOperator(o)
                            p_dom_op = col.ProjectOperatorDominant(o)
                            fact_ops[i].append(p_op)
                            fact_vals[i].append(p_v)
                            if p_dom_op in common.PROJECT_OPERATORS_DOMINANT.values(
                            ):
                                fact_dominant_ops[i].append(p_dom_op)
                            else:
                                fact_dominant_ops[i].append(None)

    assert len(fact_cols) == len(fact_ops) == len(fact_vals) == len(
        fact_dominant_ops)
    return fact_cols, fact_ops, fact_vals, fact_dominant_ops


class Aggregation(Enum):
    count = 0
    distinct_count = 1
    sum = 2
    distinct_sum = 3
    avg = 4
    distinct_avg = 5
    # TODO: distinct时，使用p(random_variable|conditions)作为筛选条件，筛除概率为零的取值，然后对剩下的取值直接计算count、sum或avg


class PredicateTreeNode:
    def __init__(self, father, left_child, right_child, op_type: str):
        self.father = father
        # ConditionTreeNode or Predicate
        self.left_child = left_child
        # if left is Probability and right is None, meaning that there is only one condition without 'and' or 'or'
        self.right_child = right_child
        # 'and', 'or'
        self.op_type = op_type

        self.predicate_list = []


class Predicate:
    def __init__(self, table: str, attr: Union[str, None], raw_attr: Union[list, None], predicate: list=None, val: list=None,
                 original_predicate: Union[list, None] = None, original_val: Union[list, None] = None,
                 raw_predicate: Union[list, None] = None, raw_val: Union[list, None] = None):
        self.table = table
        self.attr = attr # 对于key列，此处和raw_attr一样
        self.raw_attr = raw_attr
        self.fact_predicate = predicate
        self.fact_val = val
        assert (predicate is None and val is None) or len(predicate) == len(val) == 1 or len(predicate) == len(val) == 2
        self.predicate = torch.zeros((1, 2, 5)) if predicate is not None else None
        self.val = torch.zeros((1, 2, 1)) - 1 if val is not None else None
        if predicate is not None:
            for i, pred in enumerate(predicate):
                # 只有一个谓词的默认填充到前面
                if pred != -1:
                    # one_hot encoding for existing predicate
                    self.predicate[:, i, pred] = 1
            for i, v in enumerate(val):
                # 只有一个值的默认填充到前面
                if v is not None:
                    # one_hot encoding for existing predicate
                    self.val[:, i, :] = v

        self.original_predicate = original_predicate if original_predicate is not None else None
        self.original_val = original_val if original_val is not None else None

        self.raw_predicate = raw_predicate if raw_predicate is not None else None
        self.raw_val = raw_val if raw_val is not None else None

    def create_array_of_list(self, x: list):
        res = np.empty(shape=(len(x),), dtype=object)
        for i in range(len(x)):
            res[i] = x[i]
        return res

    def __eq__(self, other):
        def cmp_array_or_None(a: Union[np.ndarray, None, torch.Tensor], b: Union[np.ndarray, None, torch.Tensor]):
            return (a is None and b is None) or (
                    a is not None and b is not None and a.shape == b.shape and (a == b).all())

        if not isinstance(other, Predicate):
            return False
        else:
            return self.table == other.table and \
                self.attr == other.attr and \
                ((self.predicate is None and other.predicate is None) or (type(self.predicate) is type(other.predicate) and self.predicate == other.predicate).all()) and \
                ((self.val is None and other.predicate is None) or (type(self.val) is type(other.val) and self.val == other.val).all()) and \
                self.original_predicate == other.original_predicate and \
                self.original_val == other.original_val and \
                self.raw_predicate == other.raw_predicate and \
                self.raw_val == other.raw_val

    def __hash__(self):
        return hash(f'table:{self.table}, attr:{self.attr}, predicate:{self.predicate}, val:{self.val}')

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


@dataclass(frozen=True)
class Query:
    # 0:count, 1:sum, 2:mean, 3:avg
    agg: Aggregation
    # natural_order
    attr: int
    join_table: List[int]
    join_type: List[int]
    join_keys: List[int]
    predicate: List[Predicate]


@dataclass(frozen=True)
class RandomVariable:
    table: int
    attr: int


class Probability:
    def __init__(self, random_variable, conditions=None):
        # [table, attr]
        self._check_random_variable(random_variable)
        self._random_variable = random_variable
        # A tree of predicate, PredicateTreeNode
        self._conditions: PredicateTreeNode = conditions

    @staticmethod
    def _check_random_variable(random_variable):
        assert isinstance(random_variable, RandomVariable)

    @property
    def random_variable(self):
        return self._random_variable

    @random_variable.setter
    def random_variable(self, value):
        self._check_random_variable(value)
        self._random_variable = value

    @property
    def conditions(self):
        return self._conditions

    @conditions.setter
    def conditions(self, value):
        self.conditions = value


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))


def get_distinct_tables(predicates: List[Predicate]):
    unique_tables = np.unique([pred.table for pred in predicates])
    return len(unique_tables)


class DirectEstimator(CardEst):

    def __init__(
            self,
            models: Dict[str, torch.nn.Module],
            tables: Dict[str, Table],
            fact_tables: Dict[str, common.FactorizedTable],
            config,
            global_distinct_values,
            device=util.get_device(),
            seed=False,
            cardinality=None,  # Skip sampling on wildcards?
            requires_grad=False,
            batch_size=1
    ):
        super(DirectEstimator, self).__init__()
        # if not requires_grad:
        #     torch.set_grad_enabled(False)
        self.batch_size = 1
        self.models = models
        self.tables_dict = tables
        self.fact_tables = fact_tables if fact_tables is not None else tables
        self.base_tables = {}
        self.requires_grad = False
        # TODO: 支持不同表不同列的join
        self.global_distinct_values = global_distinct_values
        for table_name, fact_table in fact_tables.items():
            self.base_tables[table_name] = fact_table.base_table

        self.tables = list(tables.values())
        self.config = config

        # self.tabledataset = tabledataset
        self.shortcircuit = True

        self.seed = seed
        self.device = device
        # TODO: LRU for cache
        self.cache = {}
        self.cache_true = {}

        # TODO speedup
        # self.traced_fwd = [torch.jit.trace(model.forward_with_encoded_input, model.EncodeInput(
        #             torch.zeros(config['psample'], model.nin, device=util.get_device()))) for model in self.models]
        # Inference optimizations below.

        # self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        # self.traced_encode_inputs = [model.EncodeInput for model in models]
        self.inps = {name: torch.full((1, 2, len(table.columns)), -1, device=self.device) for name, table in self.fact_tables.items()}
        self.preds = {name: torch.zeros((1, 2, 5 * len(table.columns)), device=self.device) for name, table in self.fact_tables.items()}
        self._warm_up_model() #预热以防第一次预测开销过高导致速度测量不准确
        if self.models is not None:
            for m_name, m in self.models.items():
                ANPM_param_num = np.array([np.prod(t.shape) for t in [v for k, v in m.named_parameters() if 'modulation_offset' in k]]).sum()
                print(f'{m_name} ANPM params num: {ANPM_param_num}')

    def __str__(self):
        return 'Duet_Join'

    def _set_mask_(self, model):
        for layer in model.net:
            if type(layer) == made.MaskedLinear:
                layer.masked_weight = layer.mask * layer.weight
                print('Setting masked_weight in MADE, do not retrain!')


    def get_prob_of_predicate_tree(self, predicates: List, join_tables:List[str], tables_dict: Dict[str, Table], how: str = '=',
                                   real=False):
        """
        不包含join或natural join的条件概率

        在root节点的predicate_list中形成如下形式[[predicate1,predicate2,...], [...]]
        子列表之间是析取关系，子列表内部是合取关系
        首先计算子列表内部的联合概率
        再计算析取关系分解后的概率

        :param PredicateTreeNode: 描述条件的数的根节点
        :param tables_dict: 涉及到的两个Table的实例对象
        :param num_samples: 渐进式采样的样本数目
        """
        predicate_list = predicates
        with autocast():
            prob = self.get_prob_of_predicates_from_different_table(predicate_list, join_tables, tables_dict, how,
                                                                    real=real)
            if self.config['test']['faster_version']:
                if isinstance(prob, torch.Tensor):
                    return sum(prob.cpu().numpy().tolist())
                elif isinstance(prob, np.ndarray):
                    return sum(prob.tolist())
                else:
                    return prob
            else:
                return prob.sum().cpu().item()



    @staticmethod
    def get_keep_dim(predicates):
        keep_dim = 2 if len(predicates) >= 2 and predicates[-2].raw_predicate is None else 1
        return keep_dim

    '''
    默认key值为int
    '''

    def fill_complete_values(self, local_distribution, distinct_values, table_name, how, real):
        local_distribution = local_distribution.squeeze()
        if len(local_distribution) == len(self.global_distinct_values):
            return local_distribution
        elif len(local_distribution) == len(distinct_values):
            table = self.tables_dict[table_name]
            mask = table.columns_dict[self.GetJoinKeyColumn(table)].global_discretize_distinct_values_mask
            dist = torch.zeros((len(self.global_distinct_values)), device=local_distribution.device,
                               dtype=local_distribution.dtype)
            dist[mask] = local_distribution + 1e-10 if not real and how=='=' else local_distribution
            if self.config['test']['faster_version']:
                dist = dist.cpu().numpy().astype(object)
                # dist = np.array([gmpy2.mpz(d) for d in dist])
            return dist
        else:
            raise Exception(
                f'wrong shape: old_data:{local_distribution.shape}, distinct_values:{distinct_values.shape}')

    @staticmethod
    def GetJoinKeyColumn(table):
        return JoinOrderBenchmark.GetJobLightJoinKeys()[table.name.replace('.csv', '').replace('_previous','')]

    @staticmethod
    def GetKeyPredicate(table_name, column):
        return Predicate(table_name, column, column, None, None)

    def get_prob_of_predicates_from_different_table(self, predicates: List[Predicate], join_tables: List[str],
                                                    tables_dict: Dict[str, Table],
                                                    how: str = '=', real=False, reduce=True, key_only=False):

        # 首先将所有涉及到的表分为两份
        join_tables = list(join_tables)
        tables_left = join_tables[:-1]
        tables_right = join_tables[-1:]
        # 分离来自左右两组表的谓词
        predicates_from_left = list(filter(lambda x: x.table in tables_left, predicates))
        predicates_from_right = list(filter(lambda x: x.table in tables_right, predicates))
        keys_for_left = list(filter(lambda x: x.raw_val is None, predicates_from_left))
        keys_for_right = list(filter(lambda x: x.raw_val is None, predicates_from_right))
        # 计算概率密度
        left_distinct_values = self.tables_dict[tables_left[0]].columns_dict[
            JoinOrderBenchmark.GetJobLightJoinKeys()[tables_left[0]]].all_distinct_values_gpu if len(tables_left)>0 else None
        right_distinct_values = self.tables_dict[tables_right[0]].columns_dict[
            JoinOrderBenchmark.GetJobLightJoinKeys()[tables_right[0]]].all_distinct_values_gpu if len(tables_right)>0 else None
        if not key_only:
            if len(tables_left) == 1:
                # 调整顺序始终把key列放最后
                assert len(keys_for_left)==1
                predicates_from_left.remove(keys_for_left[0])
                predicates_from_left.append(keys_for_left[0])
                if real or tables_left[0] in real_table:
                    prob_left = self.get_freq_of_predicates(tuple(predicates_from_left))
                else:
                    prob_left = self.get_prob_of_predicates(predicates_from_left, keepdim=self.get_keep_dim(predicates_from_left))
                prob_left = self.fill_complete_values(prob_left, left_distinct_values, tables_left[0], how, real)
            elif len(tables_left) > 1:
                prob_left = self.get_prob_of_predicates_from_different_table(predicates_from_left, tables_left,
                                                                             tables_dict,
                                                                             how, reduce=False, real=real)
            else:
                prob_left = 1

            if len(tables_right) == 1:
                # 调整顺序始终把key列放最后
                assert len(keys_for_right) == 1
                predicates_from_right.remove(keys_for_right[0])
                predicates_from_right.append(keys_for_right[0])
                if real or tables_right[0] in real_table:
                    prob_right = self.get_freq_of_predicates(tuple(predicates_from_right))
                else:
                    prob_right = self.get_prob_of_predicates(predicates_from_right, keepdim=self.get_keep_dim(predicates_from_right))
                prob_right = self.fill_complete_values(prob_right, right_distinct_values, tables_right[0], how, real)
            elif len(tables_right)>1:
                prob_right = self.get_prob_of_predicates_from_different_table(predicates_from_right, tables_right,
                                                                              tables_dict,
                                                                              how, reduce=False, real=real)
            else:
                prob_right = 1

            if len(tables_left) > 0 and len(tables_right)>0: # 避免单表基数估计时的错误
                if isinstance(prob_left, torch.Tensor):
                    if how == '>=':
                        prob_left = prob_left + torch.sum(prob_left, dim=-1, keepdim=True) - torch.cumsum(prob_left, dim=-1)
                    elif how == '>':
                        prob_left = torch.roll(prob_left, shifts=-1, dims=-1)  # 反向移位
                        prob_left[-1:] = 0
                        prob_left = prob_left + torch.sum(prob_left, dim=-1, keepdim=True) - torch.cumsum(prob_left, dim=-1)
                    elif how == '<=':
                        prob_left = torch.cumsum(prob_left, dim=-1)
                    elif how == '<':
                        prob_left = torch.roll(prob_left, shifts=1, dims=-1)
                        prob_left[:1] = 0
                        prob_left = torch.cumsum(prob_left, dim=-1)
                elif isinstance(prob_left, np.ndarray):
                    if how == '>=':
                        prob_left = prob_left + np.sum(prob_left, axis=-1, keepdims=True) - np.cumsum(prob_left, axis=-1)
                    elif how == '>':
                        prob_left = np.roll(prob_left, shift=-1, axis=-1)  # 反向移位
                        prob_left[-1:] = 0
                        prob_left = prob_left + np.sum(prob_left, axis=-1, keepdims=True) - np.cumsum(prob_left, axis=-1)
                    elif how == '<=':
                        prob_left = np.cumsum(prob_left, axis=-1)
                    elif how == '<':
                        prob_left = np.roll(prob_left, shift=1, axis=-1)
                        prob_left[:1] = 0
                        prob_left = np.cumsum(prob_left, axis=-1)

        if not self.config['test']['faster_version']:
            if len(tables_left) == 1:
                if real or tables_left[0] in real_table:
                    prob_key_left = self.get_freq_of_predicates(tuple(keys_for_left))
                else:
                    prob_key_left = self.get_prob_of_predicates(keys_for_left, keepdim=self.get_keep_dim(keys_for_left))
                prob_key_left = self.fill_complete_values(prob_key_left, left_distinct_values, tables_left[0], how, real)
            elif len(tables_left)>1:
                prob_key_left = self.get_prob_of_predicates_from_different_table(keys_for_left, tables_left,
                                                                                 tables_dict,
                                                                                 how, real=real, key_only=True)
            else:
                prob_key_left = 1

            if len(tables_right) == 1:
                if real or tables_right[0] in real_table:
                    prob_key_right = self.get_freq_of_predicates(tuple(keys_for_right))
                else:
                    prob_key_right = self.get_prob_of_predicates(keys_for_right, keepdim=self.get_keep_dim(keys_for_right))
                prob_key_right = self.fill_complete_values(prob_key_right, right_distinct_values, tables_right[0], how, real)

            elif len(tables_right)>1:
                prob_key_right = self.get_prob_of_predicates_from_different_table(keys_for_right, tables_right,
                                                                                  tables_dict,
                                                                                  how, real=real, key_only=True)
            else:
                prob_key_right = 1

            if len(tables_left) > 0 and len(tables_right) > 0:
                if isinstance(prob_key_left, torch.Tensor):
                    if how == '>=':
                        prob_key_left = prob_key_left + torch.sum(prob_key_left, dim=-1, keepdim=True) - torch.cumsum(prob_key_left, dim=-1)
                    elif how == '>':
                        prob_key_left = torch.roll(prob_key_left, shifts=-1, dims=-1)  # 反向移位
                        prob_key_left[-1:] = 0
                        prob_key_left = prob_key_left + torch.sum(prob_key_left, dim=-1, keepdim=True) - torch.cumsum(prob_key_left, dim=-1)
                    elif how == '<=':
                        prob_key_left = torch.cumsum(prob_key_left, dim=-1)
                    elif how == '<':
                        prob_key_left = torch.roll(prob_key_left, shifts=1, dims=-1)
                        prob_key_left[:1] = 0
                        prob_key_left = torch.cumsum(prob_key_left, dim=-1)
                elif isinstance(prob_key_left, np.ndarray):
                    if how == '>=':
                        prob_key_left = prob_key_left + np.sum(prob_key_left, axis=-1, keepdims=True) - np.cumsum(prob_key_left, axis=-1)
                    elif how == '>':
                        prob_key_left = np.roll(prob_key_left, shift=-1, axis=-1)  # 反向移位
                        prob_key_left[-1:] = 0
                        prob_key_left = prob_key_left + np.sum(prob_key_left, axis=-1, keepdims=True) - np.cumsum(prob_key_left, axis=-1)
                    elif how == '<=':
                        prob_key_left = np.cumsum(prob_key_left, axis=-1)
                    elif how == '<':
                        prob_key_left = np.roll(prob_key_left, shift=1, axis=-1)
                        prob_key_left[:1] = 0
                        prob_key_left = np.cumsum(prob_key_left, axis=-1)

        if not key_only:
            # 计算两个合取式中包含两个表的公式
            if self.config['test']['faster_version']:
                prob = prob_left * prob_right
            else:
                prob = (prob_left * prob_right) / torch.sum(prob_key_left * prob_key_right)
            # assert (prob >= 0).all() , f'prob error: {prob}'
            # assert not torch.isnan(prob).all() if isinstance(prob, torch.Tensor) else True, f'prob error: {prob}'
            # assert not torch.isinf(prob).all() if isinstance(prob, torch.Tensor) else True, f'prob error: {prob}'
            # assert prob.sum() >=0, f'prob error: {prob}'
            return prob
        else:
            assert not self.config['test']['faster_version']
                # 计算两个合取式中包含两个表的公式
            tmp = prob_key_left * prob_key_right
            prob = tmp / torch.sum(tmp)
            return prob
    # @functools.cache
    def get_freq_of_predicates(self, predicates: List[Predicate]):
        assert len(set([pred.table for pred in predicates])) == 1, 'more than one table'
        # if  frozenset([frozenset(predicates)]) in self.cache_true:
        #     return self.cache_true[frozenset([frozenset(predicates)])].to(util.get_device())

        pred_key = []
        df = self.tables_dict[predicates[0].table].data
        mask = np.ones(len(df), dtype=bool)
        for predicate in predicates:
            if predicate.raw_predicate is not None: # 跳过虚谓词
                true_predicates = predicate.raw_predicate
                true_vals = predicate.raw_val
                if true_predicates is not None:
                    for true_predicate, true_val in zip(true_predicates, true_vals):
                        col = df[predicate.raw_attr]
                        if true_predicate == "=":
                            mask &= (col == true_val)
                        elif true_predicate == ">=":
                            mask &= (col >= true_val)
                        elif true_predicate == "<=":
                            mask &= (col <= true_val)
                        elif true_predicate == ">":
                            mask &= (col > true_val)
                        elif true_predicate == "<":
                            mask &= (col < true_val)
                        else:
                            raise Exception(f'not supported predicate: {predicate.raw_attr}{true_predicate}{true_val}')
            else:
                pred_key.append(predicate)
        filtered_df = self.tables_dict[predicates[0].table].data[mask]
        if len(pred_key) == 0:
            freq = torch.as_tensor([len(filtered_df)], device=util.get_device())
        elif len(pred_key) == 1:
            key_col = self.tables_dict[pred_key[0].table].columns_dict[pred_key[0].raw_attr]
            key_values = torch.as_tensor(filtered_df[key_col.name].values, device=util.get_device())

            # 使用 torch.unique 获取频次
            unique, counts = torch.unique(key_values, return_counts=True)
            counts = counts.to(torch.float64)
            if key_col.has_none:
                isnan_mask = torch.isnan(unique)
                # 处理 NaN 逻辑
                nan_count = counts[isnan_mask].sum()
                counts = counts[~isnan_mask]
                unique = unique[~isnan_mask]

            # 构建最终频次数组
            all_values = key_col.all_distinct_values_gpu
            freq = torch.zeros(len(all_values), device=util.get_device(), dtype=torch.float64)
            isin_mask = torch.isin(all_values, unique)
            freq[isin_mask] = counts
            if key_col.has_none:
                freq[0] = nan_count
            # freq = freq / len(tmp)
        else:
            raise Exception('more than one keep dim column, not support yet')
        if not self.config['test']['faster_version']:
            freq = freq / len(df)
            return freq
        # self.cache_true[frozenset([frozenset(predicates)])] = freq.cpu()
        freq = freq.to(torch.int64)
        return freq


    def get_probs_for_col(self, model, logits, natural_idx):
        """Returns probabilities for column i given model and logits."""
        with autocast():
            logits_i = model.logits_for_col(natural_idx, logits)
        return torch.softmax(logits_i, -1)

    def _warm_up_model(self):
        """模型预热方法"""
        if self.models is None:
            return

        for table_name, model in self.models.items():
            # 生成典型输入样例（需要覆盖实际使用场景的shape）
            warmup_inp = self.inps[table_name]
            warmup_pred = self.preds[table_name]

            # 执行3次预热推理（通常1-3次即可）
            with torch.no_grad():
                for _ in range(3):
                    _ = model([warmup_inp, warmup_pred]) if not isinstance(model, TesseractTransformer) \
                        else model(warmup_pred, warmup_inp)

        # 清理GPU缓存（避免预热内存影响正式运行）
        # if 'cuda' in str(device):
        #     torch.cuda.empty_cache()
        #     torch.cuda.synchronize()

    def get_prob_of_predicates(self, predicates: List[Predicate], keepdim=1):
        """
        使用模型推理指定谓词的概率分布
        :param predicates: 谓词的列表
        :param keepdim:
        keep_dim==1,保留倒数第一个谓词所属的维度（规定倒数第一个谓词为key），输出维度(Domain(key), )
        废弃：keep_dim==2，保留后两个谓词所属的维度，（规定倒数第二个谓词为random_variable，倒数第一个谓词为key），输出维度(Domain(random_variable), Domain(key))
        目前版本除了最后面的key的虚谓词，剩下的都是分解之后的

        :return prob (ndarray)
        """
        with autocast():
            assert keepdim == 1
            # 检查cache中是否有需要的概率张量（for debug and key dist only）
            if self.config['test']['use_cache'] and len(predicates) == 1 and frozenset([frozenset(predicates), keepdim]) in self.cache:
                return self.cache[frozenset([frozenset(predicates), keepdim])].to(util.get_device())
            table = predicates[0].table
            fact_table = self.fact_tables[table]
            fact_columns = fact_table.columns
            original_table = self.tables_dict[table]
            # 按顺序预测谓词合取式的概率
            # 规定倒数第二个谓词为random_variable，倒数第一个谓词为key
            model = self.models[table]

            cols_id_to_keep_dim = []
            sub_col_to_deal_output = set()
            col_to_deal_output = set()

            for pred in predicates[-keepdim:]:
                subvar_list = fact_table.map_from_col_to_fact_col[pred.attr]
                # key column always use natural column name
                col_to_deal_output.add(fact_table.base_table.columns_name_to_idx[pred.attr])
                for subvar in subvar_list:
                    col_id = fact_table.columns_name_to_idx[subvar]
                    # print(f'keep dim for join key {table}:{subvar}:{col_id}, nat_ordering:{nat_ordering}')
                    cols_id_to_keep_dim.append(col_id)
                    sub_col_to_deal_output.add(col_id)

            ncols = len(fact_columns)
            inp = self.inps[table]
            pred = self.preds[table]
            inp.fill_(-1)
            pred.fill_(0)
            valid_i_list = [None] * ncols  # None means all valid.
            for predicate in predicates[:-keepdim]:
                fact_col_idx = fact_table.columns_name_to_idx[predicate.attr]
                sub_col_to_deal_output.add(fact_col_idx)
                col_to_deal_output.add(fact_table.base_table.columns_name_to_idx[fact_table.map_from_fact_col_to_col[predicate.attr]])
                col_name = fact_table.map_from_fact_col_to_col[predicate.attr]
                col = fact_table.base_table.columns_dict[col_name]
                col_idx = fact_table.base_table.columns_name_to_idx[col_name]
                valid_i = None
                for op, v in zip(predicate.original_predicate, predicate.original_val):
                    # There exists a filter.
                    dvs = col.all_discretize_distinct_values_gpu
                    valid = torch_OPS[op](dvs, torch.as_tensor(v, device=self.device))
                    if op in [2,4] and col.has_none:
                        valid[0] = False # zero out nan
                    valid_i = valid if valid_i is None else torch.bitwise_and(valid_i, valid)
                valid_i_list[col_idx] = valid_i.float()

            # 启用列分解后，模型的输入的是分解后的谓词分量
            for predicate in predicates[:-keepdim]:
                # 对于非key列不分解
                col_idx = fact_table.columns_name_to_idx[predicate.attr]
                for val, op in zip(predicate.val, predicate.predicate):
                    inp[...,  col_idx:col_idx + 1] = val
                    pred[..., col_idx * 5:(col_idx + 1) * 5] = op
            logits = model([inp, pred]) if not isinstance(model, TesseractTransformer) else model(pred, inp)

            # 按fact_nat_idx存储
            fact_masked_probs_keys = {}
            output = [None for _ in range(ncols)]
            p = 1.0
            for natural_idx in range(ncols):
                if natural_idx not in sub_col_to_deal_output:
                    continue
                probs_i_raw = self.get_probs_for_col(model, logits, natural_idx+1 if model.use_sos else natural_idx).squeeze(0)
                if natural_idx in cols_id_to_keep_dim:
                    fact_masked_probs_keys[natural_idx] = probs_i_raw
                output[natural_idx] = probs_i_raw

            for natural_idx, col in enumerate(fact_table.base_table.columns):
                if fact_table.base_table.columns_name_to_idx[col.name] not in col_to_deal_output:
                    continue
                fact_cols = fact_table.map_from_col_to_fact_col[col.name]
                fact_cols_idx = [fact_table.columns_name_to_idx[fact_col] for fact_col in fact_cols]
                dist = None
                for fact_col_idx in fact_cols_idx:
                    if dist is None:
                        # [dist]
                        dist = output[fact_col_idx]
                    else:
                        # 自回归性保证asset成立（似乎由于float32精度问题不成立）
                        # assert (dist[0, :] == dist[1, :]).all()
                        dist = (dist.unsqueeze(-1) * output[fact_col_idx]).flatten() #.unsqueeze(0)
                masked_dist = dist[:col.distribution_size]
                masked_dist *= dist.sum() / masked_dist.sum()
                if valid_i_list[natural_idx] is not None:
                    valid_i = valid_i_list[natural_idx]
                    masked_dist *= valid_i
                    p *= torch.sum(masked_dist, dim=-1, keepdim=False)
                else:
                    p = p * masked_dist

            if self.config['test']['faster_version']:
                p = p * original_table.cardinality
            if self.config['test']['use_cache'] and len(predicates) == 1:
                self.cache[frozenset([frozenset(predicates), keepdim])] = p.cpu()
            return p

    def _print_probs(self, columns, operators, vals, ordering, masked_logits):
        ml_i = 0
        for i in range(len(columns)):
            natural_idx = ordering[i]
            if operators[natural_idx] is None:
                continue
            truncated_vals = self._truncate_val_string(vals[natural_idx])
            print('  P({} {} {} | past) ~= {:.8f}'.format(
                columns[natural_idx].name, operators[natural_idx],
                truncated_vals, masked_logits[i].mean().cpu().item()))
            ml_i += 1

    def _truncate_val_string(self, val):
        truncated_vals = []
        for v in val:
            if type(v) == tuple:
                new_val = str(list(v)[:20]) + '...' + str(
                    len(v) - 20) + ' more' if len(v) > 20 else list(v)
            else:
                new_val = v
            truncated_vals.append(new_val)
        return truncated_vals


def FillInUnqueriedColumns(table, columns, operators, vals):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        if isinstance(c, common.Column):
            idx = table.ColumnIndex(c.name)
        elif isinstance(c, str):
            idx = table.ColumnIndex(c)
        else:
            raise Exception(f'error type: {str(type(c))} of c')

        if os[idx] is None:
            os[idx] = [o]
            vs[idx] = [v]
        else:
            # Multiple clauses on same attribute.
            os[idx].append(o)
            vs[idx].append(v)

    return cs, os, vs
