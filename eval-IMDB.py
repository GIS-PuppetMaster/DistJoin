import argparse
import warnings

from torch.cpu.amp import autocast

import experiments
import yaml
import torch
import copy
import os.path
import re
import time
import wandb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle as pkl
import queue
import AQP_estimator
import common
import datasets
from datasets import JoinOrderBenchmark
from AQP_estimator import DirectEstimator, PredicateTreeNode, op_types, Predicate
from queries.GenerateMSCNWorkload import workload_num
from utils import util
from utils.util import MakeIMDBTables, GPUMemoryMonitor
import utils
import sys


# torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='IMDB', help='config name')
parser.add_argument('--exp_mark', type=str, default='', help='experiment mark')
parser.add_argument('--no_wandb', action="store_true", default=False, help='use wandb')
args = parser.parse_args()

with open(f'./Configs/{args.config}/{args.config}.yaml', 'r', encoding='utf-8') as f:
    raw_config = yaml.safe_load(f)
config_excludes = raw_config['excludes']
config_train = raw_config['train']
config = raw_config['test']
# assert pow(2, config_train['word_size_bits']) <= config['psample']
JOB_config = experiments.JOB_LIGHT_BASE


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          table,
                          available_cols,
                          return_col_idx=False):
    # s = table.data.iloc[rng.randint(0, 100)]
    s = table.data.iloc[rng.randint(0, table.cardinality)]

    # 确定可用列，剔除nan
    ava_cols = copy.copy(available_cols)
    vals = list(s.values)
    for v in vals:
        if v != v and vals.index(v) in ava_cols:
            ava_cols.remove(vals.index(v))
            vals[vals.index(v)] = 'not a number'

    # 修改前：
    # k = np.array(list(s.keys()))[available_cols]
    # k = np.where(k == 'Reg Valid Date')[0]
    # if isinstance(k, np.ndarray) and len(k) > 0:
    #     assert len(k) == 1
    #     # Giant hack for DMV.
    #     vals[int(k[0])] = vals[int(k[0])].to_datetime64()

    assert len(ava_cols) >= num_filters

    # 修改前：idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    # cols = np.take(all_cols, idxs)
    idxs = rng.choice(ava_cols, replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    if num_filters == len(ava_cols):
        if return_col_idx:
            return np.array([table.columns.index(col) for col in cols]), ops, vals
        return ava_cols, ops, vals

    vals = s.values[idxs]
    if return_col_idx:
        return np.array([table.columns.index(col) for col in cols]), ops, vals

    return cols, ops, vals



def RunN(tables_dict,
         estimator, min_max_q_error):
    monitor = GPUMemoryMonitor(interval=0.5)  # 创建显存监控器
    monitor.start()  # 启动监控
    tag = raw_config['tag']

    total_count_RelErr = 0
    total_q_error = 0

    selectivity_list = []
    rel_error_list = []
    q_error_list = []
    dist_error_list = []
    real = config['real']
    tc = []
    queries_job_format = utils.util.JobToQuery(config['queries_csv'])
    queries_job_format_ = []
    for idx, (involved_tables, _, _, _) in enumerate(queries_job_format):
        if len(set(JoinOrderBenchmark.GetJobLightJoinKeys().keys()).intersection(set(involved_tables))) < len(
                involved_tables):
            print('skip query since some tables are not supported')
            continue
        else:
            queries_job_format_.append(queries_job_format[idx])
    queries_job_format = queries_job_format_
    loaded_queries = utils.util.UnpackQueries(estimator.tables_dict, queries_job_format)
    num = len(loaded_queries)
    result_list = []
    how = config['how']
    workload_tag = config['queries_csv'].split('/')[-1].split('.')[0]
    true_card_list = None
    # windows doesn't support '>=' in file name, use AQP_estimator.OPS_dict[how] instead of how
    if os.path.exists(f'./queries/{workload_tag}_{how}{tag}.pkl'):
        with open(f'./queries/{workload_tag}_{how}{tag}.pkl', 'rb') as f:
            true_card_list = pkl.load(f)
    elif not config['real']:
        raise Exception('ground truth not found and not running in \'real\' mode')

    used_true_card_list = []
    for query_id, (join_tables, join_keys, preds, true_card) in enumerate(loaded_queries):
        if true_card_list:
            true_card = true_card_list[query_id]

        print(f'query_id: {query_id}\n join_tables: {join_tables}, join_keys:{join_keys}, preds:{preds}')
        if true_card == 0:
            print('true card = 0, skip')
            continue
        used_true_card_list.append(true_card)
        predicates = []
        used_table = set()
        for table_name, pred in preds.items():
            used_table.add(table_name)
            columns = pred['cols']
            operators = pred['ops']
            vals = pred['vals']
            columns, operators, vals = AQP_estimator.FillInUnqueriedColumns(estimator.base_tables[table_name], columns,
                                                                            operators, vals)
            fact_cols, fact_ops, fact_vals, fact_dominant_ops = AQP_estimator.ProjectQuery(
                estimator.fact_tables[table_name], columns, operators, vals)
            table = estimator.fact_tables[table_name]
            # if not raw_config['factorize']:
            #     table = table.base_table

            for fact_col, ops, vs in zip(fact_cols, fact_ops, fact_vals):
                if ops is None:
                    continue
                ops = list(map(lambda x: AQP_estimator.OPS_dict[x], ops))
                vs = table.columns_dict[fact_col.name].ValToBin(vs)
                col_idx = table.base_table.columns_name_to_idx[fact_col.raw_col_name]
                original_projected_ops = list(map(lambda x: AQP_estimator.OPS_dict[x], operators[col_idx]))
                original_projected_vals = [table.base_table.columns_dict[fact_col.raw_col_name].ValToBin(x) for x in vals[col_idx]]
                col_name = fact_col.name
                predicates.append(Predicate(table_name, col_name, table.map_from_fact_col_to_col[fact_col.name],
                                            ops, vs,
                                            original_projected_ops, original_projected_vals,
                                            operators[col_idx], vals[col_idx]))

        # 对于那些不含有谓词的表，插入虚谓词
        for table_name in join_tables:
            predicates.append(AQP_estimator.DirectEstimator.GetKeyPredicate(table_name, AQP_estimator.DirectEstimator.GetJoinKeyColumn(tables_dict[table_name])))
        st = time.time()
        condition_prob = estimator.get_prob_of_predicate_tree(predicates, join_tables, tables_dict, how, real=real)
        st = time.time() - st
        base_card = datasets.JoinOrderBenchmark.TRUE_JOIN_BASE_CARDINALITY[how][frozenset(join_tables)]
        pred_card = np.round(condition_prob) if config['faster_version'] else np.ceil(condition_prob*base_card)
        result_list.append(pred_card)
        print(f'prob:{condition_prob}, pred_card:{pred_card}, true_card:{true_card}, time cost: {st} s')
        tc.append(st)
        max_memory_usage = monitor.stop()
        print(f"推理完成，最大显存占用: {max_memory_usage:.4f} GB")
        # selectivity = true_card
        # condition_prob *= 不包含谓词的join后元组个数
        # if selectivity != 0:
        rel_error = np.abs(pred_card - true_card) / true_card
        #     # print(f'selectivity:{selectivity}')
        print(f'count_RelErr:{rel_error}')
        # else:
        #     count_RelErr = np.array([0])
        q_error = get_q_error(pred_card, true_card)
        print(f'q_error:{q_error}')

        dist_error = pred_card/true_card if pred_card > true_card else -true_card/pred_card

        selectivity_list.append(true_card / base_card)
        rel_error_list.append(rel_error)
        q_error_list.append(q_error)
        dist_error_list.append(dist_error)

        total_count_RelErr += rel_error
        total_q_error += q_error
        # with open(f'./Configs/{args.config}/{args.config}_query_ground_truth_seed{seed}.json', 'w',
        #           encoding='utf-8') as f:
        #     json.dump(ground_truth_cache, f)

    summary = {f'count RelErr': total_count_RelErr / num,
               f'q error': total_q_error / num,
               f'mean time cost': np.array(tc).mean()}
    print(summary)
    selectivity_list = np.array(selectivity_list)

    rel_error_list = np.array(rel_error_list)
    rel_error_high = rel_error_list[selectivity_list > 0.02]
    rel_error_medium = rel_error_list[np.bitwise_and(selectivity_list > 0.005, selectivity_list <= 0.02)]
    rel_error_low = rel_error_list[selectivity_list < 0.005]
    print(f'Rel-Error')
    print('\tselectivity high((2%,100%])')
    rel_high_df = pd.DataFrame(rel_error_high)
    print(
        f'\tMean: {rel_high_df.mean().to_numpy().item()}, 50th: {rel_high_df.quantile(q=0.5).to_numpy().item()}, 95th: {rel_high_df.quantile(q=0.95).to_numpy().item()}, 99th: {rel_high_df.quantile(q=0.99).to_numpy().item()}, max:{rel_high_df.max().to_numpy().item()}')
    print('\tselectivity medium((0.5%,2%])')
    rel_medium_df = pd.DataFrame(rel_error_medium)
    print(
        f'\tMean: {rel_medium_df.mean().to_numpy().item()}, 50th: {rel_medium_df.quantile(q=0.5).to_numpy().item()}, 95th: {rel_medium_df.quantile(q=0.95).to_numpy().item()}, 99th: {rel_medium_df.quantile(q=0.99).to_numpy().item()}, max:{rel_medium_df.max().to_numpy().item()}')
    print('\tselectivity low((0%,0.5%])')
    rel_low_df = pd.DataFrame(rel_error_low)
    print(
        f'\tMean: {rel_low_df.mean().to_numpy().item()}, 50th: {rel_low_df.quantile(q=0.5).to_numpy().item()}, 95th: {rel_low_df.quantile(q=0.95).to_numpy().item()}, 99th: {rel_low_df.quantile(q=0.99).to_numpy().item()}, max:{rel_low_df.max().to_numpy().item()}')
    rel_df = pd.DataFrame(rel_error_list)
    print(
        f'Rel-Error: 50th: {rel_df.quantile(q=0.5).to_numpy().item()}, 95th: {rel_df.quantile(q=0.95).to_numpy().item()}, 99th: {rel_df.quantile(q=0.99).to_numpy().item()}, max:{rel_df.max().to_numpy().item()}')
    # print(q_error_list)
    q_error_list = np.array(q_error_list)
    q_error_high = q_error_list[selectivity_list > 0.02]
    q_error_medium = q_error_list[np.bitwise_and(selectivity_list > 0.005, selectivity_list <= 0.02)]
    q_error_low = q_error_list[selectivity_list < 0.005]
    print(f'Q-Error')
    print('\tselectivity high((2%,100%])')
    q_high_df = pd.DataFrame(q_error_high)
    print(
        f'\tMean: {q_high_df.mean().to_numpy().item()}, 50th: {q_high_df.quantile(q=0.5).to_numpy().item()}, 95th: {q_high_df.quantile(q=0.95).to_numpy().item()}, 99th: {q_high_df.quantile(q=0.99).to_numpy().item()}, max:{q_high_df.max().to_numpy().item()}')
    print('\tselectivity medium((0.5%,2%])')
    q_medium_df = pd.DataFrame(q_error_medium)
    print(
        f'\tMean: {q_medium_df.mean().to_numpy().item()}, 50th: {q_medium_df.quantile(q=0.5).to_numpy().item()}, 95th: {q_medium_df.quantile(q=0.95).to_numpy().item()}, 99th: {q_medium_df.quantile(q=0.99).to_numpy().item()}, max:{q_medium_df.max().to_numpy().item()}')
    print('\tselectivity low((0%,0.5%])')
    q_low_df = pd.DataFrame(q_error_low)
    print(
        f'\tMean: {q_low_df.mean().to_numpy().item()}, 50th: {q_low_df.quantile(q=0.5).to_numpy().item()}, 95th: {q_low_df.quantile(q=0.95).to_numpy().item()}, 99th: {q_low_df.quantile(q=0.99).to_numpy().item()}, max:{q_low_df.max().to_numpy().item()}')
    q_df = pd.DataFrame(q_error_list)
    prec_summary = {
        'mean q_error': np.round(q_df.mean().item(),3).item(),
        '50th q_error': np.round(q_df.quantile(q=0.5).to_numpy().item(),3).item(),
        '95th q_error': np.round(q_df.quantile(q=0.95).to_numpy().item(),3).item(),
        '99th q_error': np.round(q_df.quantile(q=0.99).to_numpy().item(),3).item(),
        '100th q_error': np.round(q_df.max().to_numpy().item(),3).item()
    }
    os.makedirs('./result/',exist_ok=True)
    print(prec_summary)
    summary.update(prec_summary)
    if config['real']:
        # windows doesn't support '>=' in file name, use AQP_estimator.OPS_dict[how] instead of how
        with open(f'./queries/{workload_tag}_{how}{tag}.pkl', 'wb') as f:
            pkl.dump(result_list, f)

    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    ax = axes[0]
    ax.set_ylabel('q_error')
    ax.set_yscale('log')
    ax.set_xlabel(workload_tag)
    ax.boxplot(q_error_list, meanline=True, showmeans=True, showfliers=True, sym='^')
    ax = axes[1]
    upper_bound = int(np.ceil(np.log10(np.array(dist_error_list).max())))
    min_dist_error = np.array(dist_error_list).min()
    lower_bound = int(np.ceil(np.log10(np.abs(min_dist_error))))
    if min_dist_error<0:
        lower_bound = -lower_bound
    bins = np.linspace(lower_bound, upper_bound, num=upper_bound - lower_bound + 1)
    neg_mask = bins < 0
    bins = np.power(10, np.abs(bins))
    bins[neg_mask] = -bins[neg_mask]
    # bins = [-np.inf, -1e6, -1e5,-1e4,-1e3,-1e2,-1e1,0,1e1,1e2,1e3,1e4,1e5,1e6, np.inf]
    counts, bin_edges = np.histogram(dist_error_list, bins=bins)
    bin_edges = np.log10(np.abs(bin_edges))
    bin_edges[neg_mask] = -bin_edges[neg_mask]
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge')
    ax.set_ylabel('query number')
    ax.set_xlabel('q_error, log10')
    ax = axes[2]
    ax.set_ylabel('q_error')
    ax.set_xlabel(workload_tag)
    ax.boxplot(q_error_list, meanline=True, showmeans=True, showfliers=True, sym='^')
    if q_df.max().to_numpy().item() < min_max_q_error:
        fig.savefig('best_max_q_error.png')
        q_df.to_csv('best_max_q_error.csv')
        # csv_res = pd.DataFrame({'true_card': used_true_card_list, 'est_card': result_list, 'selectivity': selectivity_list, 'q_error': q_error_list, 'rel_error': rel_error_list, 'cost': tc})
        # csv_res.to_csv(f'./result/{exp_mark}_{workload_tag}_{how}{tag}.csv')
    if not sys.flags.debug:
        if not args.no_wandb:
            summary['box_plot'] = wandb.Image(fig)
            summary['q_errors'] = wandb.Histogram(q_error_list, num_bins=min(256, len(q_error_list)))
        summary['max_q_error'] = q_df.max().to_numpy().item()
        if not args.no_wandb:
            wandb.log(summary)
    plt.title(str(prec_summary))
    plt.show()
    return q_df.max().to_numpy().item()


def SaveEstimators(path, estimators, return_df=False):
    # name, query_dur_ms, errs, est_cards, true_cards
    # TODO: modify
    results = pd.DataFrame()
    for est in estimators:
        data = {
            'est': [est.name] * len(est.errs),
            'err': est.errs,
            'est_card': est.est_cards,
            'true_card': est.true_cards,
            'query_dur_ms': est.query_dur_ms,
        }
        results = results.append(pd.DataFrame(data))
    if return_df:
        return results
    results.to_csv(path, index=False)


def get_join_table_cardinality(tables_dict):
    # 默认连接key一致
    cardinality = {}
    distinct_values = set()
    tables_name = JoinOrderBenchmark.GetJobLightJoinKeys().keys()
    join_tables = list(map(lambda x: tables_dict[x], tables_name))
    for table in join_tables:
        join_key = table.columns_dict[JoinOrderBenchmark.GetJobLightJoinKeys()[table.name]]
        distinct_values = distinct_values.union(set(join_key.all_distinct_values))
    distinct_values = np.sort(np.array(list(distinct_values)), axis=0)
    all_tables_name = set()
    with open(config['queries_csv']) as f:
        for line in f.readlines():
            tables = line.strip().split("#")[0].split(",")
            tables_name = []
            for t in tables:
                tables_name.append(t.split(" ")[0])
            all_tables_name.add(tuple(tables_name))
    table_key_freq = {}
    for tables_name in list(all_tables_name):
        temp_list = None
        total_name = ' '.join(tables_name)
        join_tables = list(map(lambda x: tables_dict[x], tables_name))
        for table in join_tables:
            if not os.path.exists(f'./Configs/{args.config}/cache/table_{table.name}_freq.pkl'):
                join_key = table.columns_dict[JoinOrderBenchmark.GetJobLightJoinKeys()[table.name]]
                table_values, counts = np.unique(join_key.data.values, return_counts=True)
                test = np.isin(distinct_values, table_values, assume_unique=True)
                table_key_freq[table] = np.zeros(distinct_values.size)
                table_key_freq[table][test] = counts
                with open(f'./Configs/{args.config}/cache/table_{table.name}_freq.pkl', 'wb') as f:
                    pkl.dump(table_key_freq[table], f)
            else:
                with open(f'./Configs/{args.config}/cache/table_{table.name}_freq.pkl', 'rb') as f:
                    table_key_freq[table] = pkl.load(f)
        for i in range(len(join_tables) - 1):
            if i == 0:
                temp_list = table_key_freq[join_tables[0]] * table_key_freq[join_tables[1]]
            else:
                temp_list *= table_key_freq[join_tables[i + 1]]
        cardinality[total_name] = int(np.sum(temp_list).item())
        print(f'{tables_name}: {cardinality[total_name]},')
        with open(f'./Configs/{args.config}/cache/natural_join_cardinality.pkl', 'wb') as f:
            pkl.dump(cardinality, f)
        # print(cardinality[total_name])
    # print(cardinality)


def build_estimator(exp_mark, multi):
    models = {}
    fact_tables = {}
    # Load Tables
    tables_name = [name.replace('.csv', '') for name in JoinOrderBenchmark.GetJobLightJoinKeys().keys()]
    print(tables_name)
    use_cols = config_train['use_cols']
    tag = ''
    if raw_config['test']['load_from_cache'] and os.path.exists(
            f'./Configs/{args.config}/cache/tables_dict_{tables_name}_{use_cols}{tag}.pkl'):
        with open(f'./Configs/{args.config}/cache/tables_dict_{tables_name}_{use_cols}{tag}.pkl', 'rb') as f:
            tables_dict = pkl.load(f)
        for table_name, table in tables_dict.items():
            for col in table.columns:
                if isinstance(col.all_distinct_values_gpu, torch.Tensor):
                    col.all_distinct_values_gpu = col.all_distinct_values_gpu.to(util.get_device())
    else:
        tables_dict = MakeIMDBTables(tables_name, data_dir=raw_config['data_dir'], use_cols=use_cols, tag=tag)
        os.makedirs(f'./Configs/{args.config}/cache/', exist_ok=True)
        with open(f'./Configs/{args.config}/cache/tables_dict_{tables_name}_{use_cols}{tag}.pkl', 'wb') as f:
            pkl.dump(tables_dict, f)
    # Load Models
    # get_join_table_cardinality(tables_dict)
    global_distinct_values = set()
    for dataset in tables_name:
        if multi:
            glob = config['glob'].format(dataset, exp_mark)
        else:
            glob = config['glob'].format(dataset)
        table = tables_dict[dataset]
        key_col = table.columns_dict[JoinOrderBenchmark.GetJobLightJoinKeys()[dataset]]
        dvs = key_col.all_distinct_values
        if key_col.has_none:
            dvs = np.delete(dvs, 0)
        global_distinct_values = global_distinct_values.union(set(dvs))
        # assert raw_config['factorize']
        factorize = raw_config['factorize']
        if raw_config['test']['load_from_cache'] and os.path.exists(
                f'./Configs/{args.config}/cache/train_data_{dataset}_{factorize}{tag}.pkl'):
            with open(f'./Configs/{args.config}/cache/train_data_{dataset}_{factorize}{tag}.pkl', 'rb') as f:
                train_data = pkl.load(f)
            for col in train_data.columns:
                if isinstance(col.all_distinct_values_gpu, torch.Tensor):
                    col.all_distinct_values_gpu = col.all_distinct_values_gpu.to(util.get_device())
                if isinstance(col.all_discretize_distinct_values_gpu, torch.Tensor):
                    col.all_discretize_distinct_values_gpu = col.all_discretize_distinct_values_gpu.to(util.get_device())
        else:
            train_data = table
            if factorize:
                train_data = common.FactorizedTable(common.TableDataset(train_data, bs=config_train['bs']),
                                                    word_size_bits=config_train['word_size_bits'],
                                                    factorize_blacklist=raw_config['factorize_blacklist'])
            with open(f'./Configs/{args.config}/cache/train_data_{dataset}_{factorize}{tag}.pkl', 'wb') as f:
                pkl.dump(train_data, f)
        fact_tables[dataset] = train_data
        z = re.match('.+seed([\d\.]+).*.pt', glob)
        assert z
        seed = int(z.group(1))
        torch.manual_seed(seed)
        for i in range(len(table.columns)):
            print(table.columns[i])
            print(table.columns[i].DistributionSize())
        model = util.MakeModel(table, train_data, config, raw_config)
        model.eval()
        ReportModel(model)
        if config_train['use_data_parallel']:
            model = DataParallelPassthrough(model)

        # print('Loading ckpt:', glob)
        # model.load_state_dict(
        #     torch.load(f'./Configs/{args.config}/model/' + glob, map_location=train_utils.get_device()))
        # # model.load_state_dict(torch.load(f'./Configs/{args.config}/' + glob))
        # model.double()
        models[dataset] = model
        print(glob, seed)
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
    estimator = DirectEstimator(models,
                                tables_dict,
                                fact_tables,
                                global_distinct_values=global_distinct_values,
                                config=raw_config,
                                device=util.get_device())
    return tables_dict, fact_tables, estimator, seed


def get_q_error(est_card, card):
    if card == 0 and est_card != 0:
        return 1.0 # illegal query, will be dropped later, won't affect the result
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def Main(exp_mark):
    workload_name = os.path.split(config['queries_csv'])[-1].split('.')[0]
    tag = raw_config['tag']
    JoinOrderBenchmark.LoadTrueBaseCard(tag)
    if not sys.flags.debug and not args.no_wandb:
        wandb.login()
        wandb.init(project='AQP', id=config['how'] + exp_mark + '_' + workload_name + '_eval' if exp_mark != '' else config['how'] + exp_mark + '_' + workload_name + '_real', config=raw_config, save_code=True, job_type='test', tags=['test', ], name='test',
                   group=exp_mark if exp_mark != '' else 'real')

    gpu_set = list(range(raw_config['num_gpu']))
    for gpu_id in raw_config['exclude_gpus']:
        gpu_set.remove(gpu_id)
    util.gpu_id = gpu_set[0]
    util.gpu_id = 0
    # save = True
    # if raw_config['test']['real'] or args.no_wandb:
    #     save = False
    multi = True
    while True:
        # try:
        tables_name = [name for name in JoinOrderBenchmark.GetJobLightJoinKeys().keys()]
        tables_dict, fact_tables, estimator, seed = build_estimator(exp_mark, multi)
        min_max_q_error = float('inf')
        break
        # except Exception as e:
        #     print(e)
    mtimes = dict([(dataset, -1) for dataset in tables_name])
    while True:
        update_due_to = None
        if not config['real'] and not config['randomize']:
            loaded = False
            try:
                for dataset in tables_name:
                    glob = config['glob'].format(dataset)
                    mtime = os.path.getmtime(f'./Configs/{args.config}/model/{exp_mark}/' + glob)
                    if mtime != mtimes[dataset]:
                        print('Loading ckpt:', glob)
                        from model import made
                        # 兼容旧版的模型参数
                        param = torch.load(f'./Configs/{args.config}/model/{exp_mark}/' + glob, map_location=util.get_device())
                        for key in list(param.keys()):  # 转换为列表避免 RuntimeError
                            if 'multi_pred_embed_nn' in key:
                                del param[key]
                        try:
                            estimator.models[dataset].ConvertToUnensemble()
                            estimator.models[dataset].load_state_dict(param)
                            # estimator.update_cache()
                            estimator.models[dataset].eval()
                            if isinstance(estimator.models[dataset], made.MADE):
                                estimator.models[dataset].ConvertToEnsemble()
                        except Exception as e:
                            if isinstance(estimator.models[dataset], made.MADE):
                                estimator.models[dataset].ConvertToEnsemble()
                            estimator.models[dataset].load_state_dict(param)
                            # estimator.update_cache()
                            estimator.models[dataset].eval()

                        mtimes[dataset] = mtime
                        loaded = True
                        update_due_to = dataset
            except Exception as e:
                print(e)
                time.sleep(10)
                continue
            if not loaded:
                continue

        estimator.cache = {}
        q_error_max = RunN(tables_dict, estimator, min_max_q_error)
        print(f'q_error_max: {q_error_max}, min_max_q_error: {min_max_q_error}, update due to: {update_due_to}')
        if config['real']:
            break
        if q_error_max < min_max_q_error and multi:
            min_max_q_error = q_error_max
            # if save:
            #     for dataset in tables_name:
            #         PATH = './Configs/{}/model/{}-seed{}-{}.pt'.format(args.config, dataset, seed, 'best')
            #         print(f'save best model to {PATH}')
            #         os.makedirs(os.path.dirname(PATH), exist_ok=True)
            #         torch.save(estimator.models[dataset].state_dict(), PATH)
        if not multi:
            break

    # SaveEstimators(args.err_csv, estimator)
    # print('...Done, result:', args.err_csv)


if __name__ == '__main__':
    if not config['real'] and not config['randomize']:
        if args.exp_mark != '':
            exp_mark = args.exp_mark
        else:
            exp_mark = input('input exp mark:')
    else:
        exp_mark = ''
    Main(exp_mark.strip())
    if not sys.flags.debug and not args.no_wandb:
        wandb.finish()
