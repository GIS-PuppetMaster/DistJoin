"""Utility functions."""

import ast
import csv
from collections import defaultdict
import os
import re
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import threading
from typing import Optional
import pynvml
import time

gpu_id = 0

def get_device():
    return torch.device(f'cuda:{gpu_id}') if gpu_id >= 0 and torch.cuda.is_available() else 'cpu'


# get argmax independently along dim=2 (ncol dim)
# independent_argmax = torch.vmap(torch.argmax, in_dims=2, out_dims=-1)
# independent_argsort = torch.vmap(torch.argsort, in_dims=2, out_dims=-1)
# index = lambda x, idx: x[idx]
# indepent_index_data = torch.vmap(torch.vmap(index, in_dims=(-1, -1), out_dims=-1), in_dims=(0, 0), out_dims=0)



class GPUMemoryMonitor:
    def __init__(self, gpu_index: int = 0, interval: float = 1.0):
        """
        初始化GPU显存监控器
        :param gpu_index: 监控的GPU索引（默认0）
        :param interval: 监控间隔时间（秒）
        """
        self.gpu_index = gpu_index
        self.interval = interval
        self.max_memory_usage_gb: float = 0.0
        self._monitor_thread: Optional[threading.Thread] = None
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self.basic_memory_usage_gb = 0 # self.get_memory_GB()
        self._stop_event = threading.Event()

    def get_memory_GB(self):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        current_memory_usage_gb = mem_info.used / (1024 ** 3)  # 转换为GB
        return current_memory_usage_gb

    def _monitor_memory_usage(self):
        """后台线程：监控显存使用情况"""
        try:
            while not self._stop_event.is_set():
                current_memory_usage_gb = self.get_memory_GB() - self.basic_memory_usage_gb
                if current_memory_usage_gb > self.max_memory_usage_gb:
                    self.max_memory_usage_gb = current_memory_usage_gb
                time.sleep(self.interval)
        finally:
            pynvml.nvmlShutdown()

    def start(self):
        """启动显存监控"""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_memory_usage, daemon=True)
            self._monitor_thread.start()

    def stop(self) -> float:
        """
        停止显存监控，并返回记录的最大显存占用（GB）
        :return: 最大显存占用（GB）
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_event.set()
            self._monitor_thread.join()  # 等待监控线程结束
        return self.max_memory_usage_gb

    def clear_max_memory_usage(self):
        """清空记录的最大显存占用"""
        self.max_memory_usage_gb = 0.0

# def sortPredicates(data: torch.Tensor, preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     @param data: [bs, 2, ncol]
#     @param preds: [bs, 2, 5*ncol]
#     @return: [bs, 2, ncol], [bs, 2, 5*ncol]
#
#     """
#     # rearrange predicates order to give a fixed and specific order of predicate operator
#     # [bs, 2, 5*ncol] -> [bs, 2, ncol, 5]
#     preds_ = preds.reshape(preds.shape[0], preds.shape[1], -1, 5)
#     # [bs, 2, ncol]
#     argmax_res = independent_argmax(preds_, dim=-1)
#     # get the mask of the zero-vector in preds_
#     # [bs, 2, ncol]
#     mask = preds_.sum(dim=-1) == 0
#     # set the argmax res of zero-vector to -1 to make the argsort result correct
#     # [bs, 2, ncol]
#     argmax_res[mask] = -1
#     # stable argsort seems safer(performance drop warning)
#     # [bs, 2, ncol]
#     idx = independent_argsort(argmax_res, dim=-1)
#     # sort from big to small to make the zero-vector (empty predicate) to be at the last
#     # [bs, 2, ncol]
#     idx = torch.flip(idx, dims=[1, ])
#     # preds_.shape = [bs, 2, ncol, 5], idx.shape = [bs, 2, ncol], process along dim=(-2,-1) (ncol)
#     # and output the result to dim=2 (ncol)
#     indepent_index = torch.vmap(torch.vmap(index, in_dims=(-2, -1), out_dims=-2), in_dims=(0, 0), out_dims=0)
#     new_pred = indepent_index(preds_, idx)
#     # [bs, 2, ncol, 5] -> [bs, 2, 5*ncol]
#     preds = new_pred.flatten(2, 3)
#     # [bs, 2, ncol] ([bs, 2, ncol])-> [bs, 2, ncol]
#     data = indepent_index_data(data, idx)
#     return data, preds


def MakeOrdering(table, config, raw_config):
    fixed_ordering = None
    if ('imdb' not in raw_config['datasets'] and 'IMDB' not in raw_config['datasets']) and config[
        'special_orders'] <= 1:
        fixed_ordering = list(range(len(table.columns)))

    if config['order'] is not None:
        print('Using passed-in order:', config['order'])
        fixed_ordering = config['order']

    if config['order_seed'] is not None:
        if config['order_seed'] == 'reverse':
            fixed_ordering = fixed_ordering[::-1]
        else:
            rng = np.random.RandomState(config['order_seed'])
            rng.shuffle(fixed_ordering)
        print('Using generated order:', fixed_ordering)
    return fixed_ordering

def MakeIMDBTables(dataset, data_dir, use_cols, tag=''):
    import datasets
    return datasets.LoadImdb(dataset, data_dir=data_dir, use_cols=use_cols, try_load_parsed=False, tag=tag)

def get_key_col_name(factor_table):
    import datasets
    key_col_name = datasets.JoinOrderBenchmark.GetJobLightJoinKeys()[factor_table.base_table.name]
    return key_col_name, factor_table.map_from_col_to_fact_col[key_col_name]

def unique(items):
    """
    保序去除重复元素
    """
    seen = set()  # 集合set是一个无序不重复元素集
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)

def MakeMade(
        scale,
        layers,
        cols_to_train,
        seed,
        factor_table=None,
        fixed_ordering=None,
        natural_ordering=True,
        special_orders=0,
        inv_order=True,
        residual=True,
        direct_io=True,
        input_encoding='embed',
        output_encoding='embed',
        embed_size=32,
        dropout=True,
        fixed_dropout_ratio=False,
        input_no_emb_if_leq=False,
        embs_tied=True,
        dmol_cols=[],
        use_ensemble=False,
        use_adaptive_temp=True,
        use_ANPM=True,
        use_sos=True,
        use_mix_act=True):
    from model import made
    dmol_col_indexes = []
    if dmol_cols:
        for i in range(len(cols_to_train)):
            if cols_to_train[i].name in dmol_cols:
                dmol_col_indexes.append(i)
    # the MADE is built based on out_cols_to_train,
    # and a mapping net will map the in_cols(original columns) to out_cols(factorized columns)
    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
                     layers if layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        num_masks=max(1, special_orders),
        natural_ordering=natural_ordering,
        input_bins=[c.DistributionSize() for c in cols_to_train],
        do_direct_io_connections=direct_io,
        input_encoding=input_encoding,
        output_encoding=output_encoding,
        embed_size=embed_size,
        input_no_emb_if_leq=input_no_emb_if_leq,
        embs_tied=embs_tied,
        residual_connections=residual,
        factor_table=factor_table,
        seed=seed,
        fixed_ordering=fixed_ordering,
        dropout_p=dropout,
        fixed_dropout_p=fixed_dropout_ratio,
        use_ensemble=use_ensemble,
        use_adaptive_temp=use_adaptive_temp,
        use_ANPM=use_ANPM,
        use_sos=use_sos,
        use_mix_act=use_mix_act
    ).to(get_device())
    return model


def MakeTesseractTransformer(
        fact_table,
        d_model:int=128,
        nhead:int=4,
        num_layers:int=3
):
    from model.TesseractTransformer import TesseractTransformer
    ncol = len(fact_table.base_table.columns)
    # n_factcol = len(fact_table.columns)
    order = np.arange(ncol).tolist()
    # 把key列放最后
    from datasets import JoinOrderBenchmark
    key_col_name = JoinOrderBenchmark.GetJobLightJoinKeys()[fact_table.base_table.name]
    key_col_idx = fact_table.base_table.columns_name_to_idx[key_col_name]
    current_key_order_idx = order.index(key_col_idx)
    order.pop(current_key_order_idx)
    order.append(key_col_idx)

    ori_columns_domain_size = [c.DistributionSize() for c in fact_table.base_table.columns]
    fact_columns_domain_size = [c.DistributionSize() for c in fact_table.columns]
    num_facts_per_col = [fact_col_nums for fact_col_nums in fact_table.fact_col_number]

    bit_widths = [c.bit_width for c in fact_table.columns]
    bit_offsets = [c.bit_offset for c in fact_table.columns]
    distributions = [col.freq for i, col in enumerate(fact_table.columns)]
    model = TesseractTransformer(
        ori_columns_domain_size,
        fact_columns_domain_size,
        num_facts_per_col,
        distributions,
        order,
        bit_widths,
        bit_offsets,
        d_model,
        nhead,
        num_layers
    )
    return model



def MakeModel(table, train_data, config, raw_config, job_id=0, table_primary_index=None):
    cols_to_train = table.columns
    if raw_config['factorize']:
        cols_to_train = train_data.columns
    special_orders = raw_config['train']['special_orders']
    fixed_ordering = MakeOrdering(table, raw_config['train'], raw_config)
    table_num_columns = table_column_types = table_indexes = None
    if raw_config['train']['model_type']=='Transformer':
        orders = []
        if special_orders > 0:
            if fixed_ordering is not None:
                orders = [fixed_ordering]
            else:
                for i in range(special_orders):
                    if raw_config['train']['natural_ordering']:
                        orders.append(np.arange(len(cols_to_train)))
                    else:
                        orders.append(
                            np.random.RandomState(i + 1).permutation(
                                np.arange(len(cols_to_train))))

            # Correct for subvar ordering.
            for i in range(special_orders):
                # This could have [..., 6, ..., 4, ..., 5, ...].
                # So we map them back into:
                # This could have [..., 4, 5, 6, ...].
                # Subvars have to be in order and also consecutive
                # Done: check invert ordering
                order = orders[i]
                for orig_col, sub_cols in train_data.map_from_col_to_fact_col.items():
                    first_subvar_index = train_data.columns_name_to_idx[sub_cols[0]]
                    from datasets import JoinOrderBenchmark
                    if orig_col == JoinOrderBenchmark.GetJobLightJoinKeys()[train_data.base_table.name]:
                        print(f'table: {train_data.base_table.name}, key column: {orig_col}, first_subvar_index:{first_subvar_index}')
                        print('Before', order)
                        # 把join key列排在最后
                        order = np.delete(order, np.argwhere(order == first_subvar_index))
                        order = np.insert(order, len(order), first_subvar_index)
                        orders[i] = order
                        print('After', order)

                    print('Before', order)
                    if len(sub_cols) > 1:
                        for j in range(1, len(sub_cols)):
                            subvar_index = train_data.columns_name_to_idx[sub_cols[j]]
                            order = np.delete(order,
                                              np.argwhere(order == subvar_index))
                            order = np.insert(
                                order,
                                np.argwhere(order == first_subvar_index)[0][0] + j,
                                subvar_index)
                        orders[i] = order
                        print('After', order)
            print('Special orders', orders)
            fixed_ordering = orders[0]
        from model import transformer
        import common
        args = {
            'num_blocks': 6,
            'd_ff': 512,
            'd_model': 128,
            'num_heads': 6,
            'nin': len(cols_to_train),
            'input_bins': [c.distribution_size for c in cols_to_train],
            'use_positional_embs': True if transformer.MASK_SCHEME == 1 else False,
            'activation': 'gelu',
            'fixed_ordering': fixed_ordering,
            'dropout': raw_config['train']['dropout'],
            'per_row_dropout': False,
            'factor_table': None,
            'seed': None,
            'join_args': {
                'num_joined_tables': None,
                'table_dropout': False,
                'table_num_columns': table_num_columns,
                'table_column_types': table_column_types,
                'table_indexes': table_indexes,
                'table_primary_index': table_primary_index,
            },
            'use_ensemble': config['use_ensemble'],
        }
        args.update(raw_config['train']['transformer'])
        model = transformer.Transformer(**args).to(get_device())
    elif raw_config['train']['model_type']=='MADE':
        model = MakeMade(
            scale=raw_config['train']['fc_hiddens'],
            layers=raw_config['train']['layers'],
            cols_to_train=cols_to_train,
            seed=raw_config['seed'],
            factor_table=train_data,
            fixed_ordering=fixed_ordering,
            natural_ordering=raw_config['train']['natural_ordering'],
            special_orders=raw_config['train']['special_orders'],
            inv_order=raw_config['train']['inv_order'],
            residual=raw_config['train']['residual'],
            direct_io=raw_config['train']['direct_io'],
            input_encoding=raw_config['train']['input_encoding'],
            output_encoding=raw_config['train']['output_encoding'],
            dropout=raw_config['train']['dropout'],
            fixed_dropout_ratio=raw_config['train']['fixed_dropout_ratio'],
            embed_size=raw_config['train']['embed_size'],
            input_no_emb_if_leq=raw_config['train']['input_no_emb_if_leq'],
            embs_tied=raw_config['train']['embs_tied'],
            dmol_cols=raw_config['train']['dmol_cols'] if raw_config['train']['num_dmol'] else [],
            use_ensemble=config['use_ensemble'],
            use_adaptive_temp= raw_config['train']['use_adaptive_temp'],
            use_ANPM= raw_config['train']['use_ANPM'],
            use_sos=raw_config['train']['use_sos'],
            use_mix_act=raw_config['train']['use_mix_act']
        ).to(get_device())
    elif raw_config['train']['model_type']=='TesseractTransformer':
        model = MakeTesseractTransformer(train_data).to(get_device())
    else:
        raise ValueError(f'Unsupported model type: {raw_config["train"]["model_type"]}')
    return model


def check_sample(table, tuples, new_tuples, new_preds):
    from estimators import torch_OPS
    num_samples = tuples.shape[0]
    nin = tuples.shape[1]
    eval_preds = torch.zeros((num_samples, 2, nin)).numpy().astype(object)
    for i in range(nin):
        assert (new_tuples[..., i] < table.columns[i].distribution_size).all()
        eval_preds[..., i] = torch_OPS[torch.argmax(new_preds[..., i * 5:(i + 1) * 5], dim=-1).cpu().numpy()]
    for i in range(num_samples):
        for c in range(new_tuples.shape[-1]):
            cv = tuples[i, c]
            # assert (new_tuples[i, 0, c] != -1).all()
            for j in range(2):
                if new_tuples[i, j, c] != -1:
                    assert eval_preds[i, j, c](cv, new_tuples[i, j, c])


def ConvertSQLQueryToBin(query):
    from AQP_estimator import OPS_dict
    cols = query[0]
    ops = query[1]
    vals = query[2]
    for j, op in enumerate(ops):
        ops[j] = OPS_dict[op]

    for j, (col, val) in enumerate(zip(cols, vals)):
        vals[j] = np.where(col.all_distinct_values == val)[0].item()


def ConvertOPSBinToSQLQuery(ops):
    from AQP_estimator import OPS_array
    n = len(ops) if isinstance(ops, list) else ops.shape[-1] // 5
    new_ops = [[None, None] for _ in range(n)]
    for i in range(n):
        op = ops[:, i * 5:(i + 1) * 5]
        if op[0].max() > 0:
            new_ops[i][0] = OPS_array[op[0].argmax().item()]
            if op[1].max() > 0:
                new_ops[i][1] = OPS_array[op[1].argmax().item()]
    return new_ops


def in2d_chunked(x, all_x, chunk_size=10000):
    # 结果数组
    result = np.empty(len(x), dtype=int)
    nan_ind = pd.isnull(x)
    nan_ind_all_x = pd.isnull(all_x)

    # 找到 `all_x` 中的 NaN 索引
    if nan_ind_all_x.any():
        nan_pos = np.argmax(nan_ind_all_x)
    else:
        nan_pos = None

    # 分块处理
    for start in range(0, len(x), chunk_size):
        end = start + chunk_size
        chunk = x[start:end]

        # 比较 chunk 与 all_x
        mask = chunk[:, None] == all_x[None, :]
        if nan_pos is not None:
            mask[nan_ind[start:end], nan_pos] = True

        result[start:end] = np.where(mask)[1]

    return result


def in2d(x, all_x):
    mask = x[:, None] == all_x[None, :]
    ind = pd.isnull(x)
    nan_ind = pd.isnull(all_x)
    if nan_ind.any():
        nan_pos = np.argmax(nan_ind)
        assert np.isnan(all_x[nan_pos])
        assert (mask[ind, :].sum(1) == 0).all()
        mask[ind, nan_pos] = True
    return np.where(mask)[1]


def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          table,
                          dataset,
                          return_col_idx=False,
                          bound=False):
    num_filters = min(num_filters, len(all_cols) - 1)
    # key column shouldn't have predicates
    # 1. 找到 'id' 或 'movie_id' 对应的列索引
    key_idx = next((k for k, col in enumerate(all_cols) if col.name in ['id', 'movie_id']), -1)

    # 2. 过滤掉 key_idx，生成候选列的索引
    remaining_cols_idx = [k for k in range(len(all_cols)) if k != key_idx]

    s = None
    escape = 0
    while s is None or len(remaining_cols_idx)-np.sum(pd.isnull(s))<num_filters:
        s = table.data.iloc[rng.randint(0, table.cardinality)]
        escape+=1
        if escape>10:
            num_filters-=1
            escape = 0
    vals = s.values

    if 'dmv' in dataset:
        # Giant hack for DMV.
        vals[6] = vals[6].to_datetime64()

    idxs = None
    while idxs is None or pd.isnull(vals[idxs]).any():
        idxs = rng.choice(remaining_cols_idx, replace=False, size=num_filters)
        cols = np.take(all_cols, idxs)

    vals = vals[idxs]
    num_filters = len(vals)
    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['=', '>', '<', '>=', '<='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
    if return_col_idx:
        return idxs, ops, vals

    return cols, ops, vals


def GenerateQuery(all_cols, rng, table, dataset, return_col_idx=False, num_filters=None, bound=False, how='='):
    """Generate a random query."""
    if num_filters is not None:
        num_filters = min(num_filters, len(table.columns))
    else:
        if dataset == 'dmv':
            if bound:
                num_filters = np.clip(int(rng.gamma(5, 2)), 1, 11)
            else:
                num_filters = rng.randint(5, 12)
        elif dataset == 'cup98':
            if bound:
                num_filters = np.clip(int(rng.gamma(10, 2)), 1, 100)
            else:
                # num_filters = np.clip(int(rng.normal(20, 2)), 1, 100)
                num_filters = rng.randint(5, 101)
        elif dataset == 'census':
            if bound:
                num_filters = np.clip(int(rng.gamma(7, 2)), 1, 13)
            else:
                num_filters = rng.randint(5, 14)
        else:
            # num_filters = rng.randint(1, min(max(2, int(len(table.columns) * 0.3)), 4)) if how=='=' else rng.randint(max(1, int(len(table.columns) * 0.3)), len(table.columns))
            num_filters = rng.randint(max(0, int(len(table.columns) * 0.5)), len(table.columns))
    if num_filters==0:
        return [[],[],[]]
    cols, ops, vals = SampleTupleThenRandom(all_cols,
                                            num_filters,
                                            rng,
                                            table,
                                            dataset=dataset,
                                            return_col_idx=return_col_idx,
                                            bound=bound)
    return [cols, ops, vals]


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def train_background(args):
    res = re.match('.*--tag=([^ ]+)[ ]', args)
    assert res
    tag = res.groups()[0]
    command = "nohup " + sys.executable + " train_model.py " + args + f" >./log/{tag}.txt" " &"
    print(command)
    os.system(command)


class EvalParam:
    def __init__(self, glob, blacklist=None, psample=2000, order=None, gpu_id=3, start_epoch=0, load_queries='',
                 inference_opts=False, use_oracle=False, load_cache=True,
                 full_eval=True, num_queries=20, query_seed=1234, dataset='dmv-tiny', multi_pred_embedding='mlp',
                 err_csv='results.csv',
                 fc_hiddens=128, layers=4, residual=False, direct_io=False, inv_order=False, input_encoding='binary',
                 output_encoding='one_hot',
                 heads=0, blocks=2, dmodel=32, dff=128, transformer_act='gelu', run_sampling=False, run_maxdiff=False,
                 run_bn=False, bn_samples=200,
                 bn_root=0, maxdiff_limit=30000, tag=None, end_epoch=100, result_tag=None, use_ensemble=False):
        self.glob = glob
        # only for building model parse, estimator generate ensemble even this is false
        self.use_ensemble = use_ensemble
        self.blacklist = blacklist
        self.psample = psample
        self.result_tag = result_tag
        self.order = order
        self.gpu_id = gpu_id
        self.start_epoch = start_epoch
        self.load_queries = load_queries
        self.inference_opts = inference_opts
        self.use_oracle = use_oracle
        self.load_cache = load_cache
        self.full_eval = full_eval
        self.num_queries = num_queries
        self.query_seed = query_seed
        self.dataset = dataset
        self.multi_pred_embedding = multi_pred_embedding
        self.err_csv = err_csv
        self.fc_hiddens = fc_hiddens
        self.layers = layers
        self.residual = residual
        self.direct_io = direct_io
        self.inv_order = inv_order
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self.heads = heads
        self.blocks = blocks
        self.dmodel = dmodel
        self.dff = dff
        self.transformer_act = transformer_act
        self.run_sampling = run_sampling
        self.run_maxdiff = run_maxdiff
        self.run_bn = run_bn
        self.bn_samples = bn_samples
        self.bn_root = bn_root
        self.maxdiff_limit = maxdiff_limit
        self.tag = tag
        self.end_epoch = end_epoch


class TrainParam:
    def __init__(self, order=None, gpu_id=0, num_queries=100, query_seed=42, dataset='dmv-tiny',
                 multi_pred_embedding='mlp', err_csv='results.csv', fc_hiddens=123, layers=4, residual=False,
                 direct_io=False, inv_order=False, input_encoding='binary', output_encoding='one_hot', heads=0,
                 blocks=2, dmodel=32, dff=128, transformer_act='gelu', tag=None,
                 use_workloads=False, independent=False, bs=1024, warmups=0, data_model_warmups=0, epochs=20,
                 constant_lr=None, num_orderings=1, q_weight=1e-2, expand_factor=4, use_ensemble=False):
        self.order = order
        self.use_ensemble = use_ensemble
        self.expand_factor = expand_factor
        self.gpu_id = gpu_id
        self.num_queries = num_queries
        self.query_seed = query_seed
        self.dataset = dataset
        self.multi_pred_embedding = multi_pred_embedding
        self.err_csv = err_csv
        self.fc_hiddens = fc_hiddens
        self.layers = layers
        self.residual = residual
        self.direct_io = direct_io
        self.inv_order = inv_order
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self.heads = heads
        self.blocks = blocks
        self.dmodel = dmodel
        self.dff = dff
        self.transformer_act = transformer_act
        self.tag = tag
        self.use_workloads = use_workloads
        self.independent = independent
        self.bs = bs
        self.warmups = warmups
        self.data_model_warmups = data_model_warmups
        self.epochs = epochs
        self.constant_lr = constant_lr
        self.num_orderings = num_orderings
        self.num_queries = num_queries
        self.q_weight = q_weight


def _get_table_dict(tables):
    table_dict = {}
    for t in tables:
        split = t.split(' ')
        if len(split) > 1:
            # Alias -> full table name.
            table_dict[split[1]] = split[0]
        else:
            # Just full table name.
            table_dict[split[0]] = split[0]
    return table_dict


def _get_join_dict(joins, table_dict, use_alias_keys):
    from ordered_set import OrderedSet
    join_dict = defaultdict(OrderedSet)
    for j in joins:
        ops = j.split('=')
        op1 = ops[0].split('.')
        op2 = ops[1].split('.')
        t1, k1 = op1[0], op1[1]
        t2, k2 = op2[0], op2[1]
        if not use_alias_keys:
            t1 = table_dict[t1]
            t2 = table_dict[t2]
        join_dict[t1].add(k1)
        join_dict[t2].add(k2)
    return join_dict


def _try_parse_literal(s):
    try:
        ret = ast.literal_eval(s)
        # IN needs a tuple operand
        # String equality needs a string operand
        if isinstance(ret, tuple) or isinstance(ret, str):
            return ret
        return s
    except:
        return s


def _get_predicate_dict(predicates, table_dict):
    predicates = [predicates[x:x + 3] for x in range(0, len(predicates), 3)]
    predicate_dict = {}
    for p in predicates:
        split_p = p[0].split('.')
        table_name = table_dict[split_p[0]]
        if table_name not in predicate_dict:
            predicate_dict[table_name] = {}
            predicate_dict[table_name]['cols'] = []
            predicate_dict[table_name]['ops'] = []
            predicate_dict[table_name]['vals'] = []
        predicate_dict[table_name]['cols'].append(split_p[1])
        predicate_dict[table_name]['ops'].append(p[1])
        predicate_dict[table_name]['vals'].append(_try_parse_literal(p[2]))
    return predicate_dict


def JobToQuery(csv_file, use_alias_keys=True):
    """Parses custom #-delimited query csv.

    `use_alias_keys` only applies to the 2nd return value.
    If use_alias_keys is true, join_dict will use aliases (t, mi) as keys;
    otherwise it uses real table names (title, movie_index).

    Converts into (tables, join dict, predicate dict, true cardinality).  Only
    works for single equivalency class.
    """
    queries = []
    with open(csv_file) as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            reader = csv.reader(row)  # comma-separated
            table_dict = _get_table_dict(next(reader))
            join_dict = _get_join_dict(next(reader), table_dict, use_alias_keys)
            predicate_dict = _get_predicate_dict(next(reader), table_dict)
            true_cardinality = int(next(reader)[0])
            queries.append((list(table_dict.values()), join_dict,
                            predicate_dict, true_cardinality))

        return queries


def UnpackQueries(table_dict, queries):
    """Converts custom query representation to (cols, ops, vals)."""
    converted = []
    for q in queries:

        tables, join_dict, predicate_dict, true_cardinality = q
        # All predicates in a query (an AND of these).
        new_pred = {}
        for table, preds in predicate_dict.items():
            query_cols, query_ops, query_vals = [], [], []
            cols = preds['cols']
            ops = preds['ops']
            vals = preds['vals']
            assert len(cols) == len(ops) and len(ops) == len(vals)
            for c, o, v in zip(cols, ops, vals):
                column = list(filter(lambda x: x.name == c, table_dict[table].Columns()))[0]
                query_cols.append(c)
                query_ops.append(o)
                # Cast v into the correct column dtype.
                cast_fn = column.all_distinct_values.dtype.type
                # If v is a collection, cast its elements.
                if isinstance(v, (list, set, tuple)):
                    qv = type(v)(map(cast_fn, v))
                else:
                    qv = cast_fn(v)
                query_vals.append(qv)
            new_pred[table] = {'cols': query_cols, 'ops': query_ops, 'vals': query_vals}
        converted.append((tables, join_dict, new_pred, true_cardinality))
    # print("converted:\n")
    # print(converted)

    return converted


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    return np.argsort(order)
    # nin = len(order)
    # inv_ordering = [None] * nin
    # for natural_idx in range(nin):
    #     inv_ordering[order[natural_idx]] = natural_idx
    # return inv_ordering


def get_dataset_tag(config):
    return '-'.join([str(config['factorize']), str(config['train']['word_size_bits']), ','.join(config['datasets']), config['train']['use_cols']])


def HumanFormat(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def drawDist(fact_table, predicates, natural_idx, probs_i, valid_i):
    import matplotlib.pyplot as plt
    import common
    from AQP_estimator import OPS, OPS_array
    c_pred = list(filter(lambda pred: fact_table.columns_name_to_idx[pred.attr] == natural_idx, predicates))[0]
    discretize_data = common.Discretize(fact_table.columns[natural_idx])
    true_selectivity = OPS[OPS_array[c_pred.original_predicate[0]]](discretize_data, c_pred.original_val[0]).astype(int).sum() / fact_table.cardinality
    pred_selectivity = np.sum(probs_i * valid_i).item()
    print("true selectivity: " + true_selectivity)
    print("predicted selectivity: " + pred_selectivity)
    print(f"error: {pred_selectivity / true_selectivity}")
    plt.plot(probs_i, label="predict")
    plt.legend()
    plt.show()
    plt.plot(np.unique(discretize_data, return_counts=True)[1] / len(discretize_data), label='data')
    plt.legend()
    plt.show()
