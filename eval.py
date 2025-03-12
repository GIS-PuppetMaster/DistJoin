import argparse
import copy
import json
import os.path
import re
import time

import numpy as np
import pandas as pd
import yaml
import torch
from matplotlib import pyplot as plt
from pandas.errors import UndefinedVariableError

import AQP_estimator
import common
import datasets
from datasets import JoinOrderBenchmark
from AQP_estimator import DirectEstimator, Probability, PredicateTreeNode, op_types, Predicate, RandomVariable
from model.made import MADE
from train import config_excludes, MakeModel

torch.set_grad_enabled(False)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device', DEVICE)

parser = argparse.ArgumentParser()

# Training.
# parser.add_argument('--config', type=str, default='DMV-tiny', help='config name')
parser.add_argument('--config', type=str, default='IMDB', help='config name')
args = parser.parse_args()

with open(f'./Configs/{args.config}/{args.config}.yaml', 'r', encoding='utf-8') as f:
    raw_config = yaml.safe_load(f)
config_excludes = raw_config['excludes']
config_train = raw_config['train']
config = raw_config['test']


def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    model = MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
                     config_train['layers'] if config_train['layers'] > 0 else [512, 256, 512, 128, 1024],
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=config_train['input_encoding'],
        output_encoding=config_train['output_encoding'],
        embed_size=32,
        seed=seed,
        do_direct_io_connections=config_train['direct_io'],
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=config_train['residual'],
        fixed_ordering=fixed_ordering,
        dropout_p=config_train['dropout'],
    ).to(DEVICE)

    return model


def MakeTable(dataset):
    table = datasets.LoadDmv(file_path=f'D:/PycharmProjects/naru/datasets/{dataset}.csv',
                             exclude=config_excludes)

    # oracle_est = estimators_lib.Oracle(table)
    if config['run_bn']:
        return table, common.TableDataset(table)
    return table, None


def MakeIMDBTables(dataset, use_cols):
    return datasets.LoadImdb(dataset, use_cols=use_cols, try_load_parsed=False)


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


def GenerateQuery(all_cols, rng, table, num_filters, available_cols, return_col_idx=False):
    """Generate a random query."""
    cols, ops, vals = SampleTupleThenRandom(all_cols,
                                            num_filters,
                                            rng,
                                            table,
                                            available_cols,
                                            return_col_idx=return_col_idx)
    return cols, ops, vals


def RunNParallel(estimator_factory,
                 parallelism=2,
                 rng=None,
                 num=20,
                 num_filters=11,
                 oracle_cards=None):
    """RunN in parallel with Ray.  Useful for slow estimators e.g., BN."""
    raise Exception('not implemented yet')


def RunN(tables,
         estimator,
         rng=None,
         num=20,
         log_every=50):
    seed = 2501
    if rng is None:
        rng = np.random.RandomState(seed)
        # rng = np.random.RandomState(np.random.randint(1,200))
    last_time = None
    total_mse = 0
    total_mae = 0
    total_avg_RelErr = 0
    total_sum_RelErr = 0
    total_count_RelErr = 0
    total_q_error = 0
    total_max_RelErr = 0
    total_min_RelErr = 0
    total_avg_num = num
    total_sum_num = num
    total_max_num = num
    total_min_num = num
    total_cross_entropy = 0
    selectivity_list = []
    count_error_list = []
    q_error_list = []
    q_error_dict = {}
    real = config['real']
    if os.path.exists(f'./Configs/{args.config}/{args.config}_query_ground_truth_seed{seed}.json'):
        with open(f'./Configs/{args.config}/{args.config}_query_ground_truth_seed{seed}.json', 'r',
                  encoding='utf-8') as f:
            ground_truth_cache = json.load(f)
    else:
        ground_truth_cache = {'selectivity': {}}
    tc = []
    for i in range(num):
        do_print = False
        if i == 2:
            print(i)
        if i % log_every == 0:
            # if last_time is not None:
            #     print('{:.1f} queries/sec'.format(log_every /
            #                                       (time.time() - last_time)))
            do_print = True
            print('Query {}:'.format(i), end=' ')
            last_time = time.time()

        root = PredicateTreeNode(None, None, None, op_types[rng.randint(2)])
        available_node = [root]
        predicates_num_lower_bound = 1
        predicates_num_higher_bound = 2
        # 可用谓词个数需要扣除重复的key
        # keys = list(map(lambda x: x.join_key_column, tables))

        keys = list(JoinOrderBenchmark.GetIndexofJobLightJoinKeys().values())

        total_attrs_num = np.sum([len(table.columns) for table in tables]) - len(keys)
        assert total_attrs_num >= predicates_num_lower_bound
        target_higher_bound = rng.randint(5, total_attrs_num)
        # 生成谓词树的节点
        # 需扣除Random Variable
        while predicates_num_higher_bound <= total_attrs_num - 1:
            # 随机选择一个节点作为父节点
            random_father = available_node[rng.randint(len(available_node))]
            new_node = PredicateTreeNode(random_father, None, None, op_types[rng.randint(2)])
            if random_father.left_child is None:
                random_father.left_child = new_node
            else:
                random_father.right_child = new_node
                available_node.remove(random_father)
                predicates_num_lower_bound += 1
            available_node.append(new_node)
            predicates_num_higher_bound += 1
            if predicates_num_higher_bound >= target_higher_bound:
                break
        # 填充谓词叶子节点
        # 首先给每个叶子节点的左子节点添加谓词，确保合法性
        available_attr = [set(range(len(t.columns))) for t in tables]

        # TODO: 允许join key做谓词，在此之前先禁止这种情况
        for table_id, key in enumerate(keys):
            if key in available_attr[table_id]:
                available_attr[table_id].remove(key)

        left_num = 0
        for node in available_node:
            if node.left_child is None:
                ts = copy.copy(tables)
                random_table = rng.randint(0, len(tables))
                ts.pop(random_table)
                have_no_attr = False
                while len(available_attr[random_table]) <= 1:
                    if len(ts) == 0:
                        have_no_attr = True
                        break
                    rt = np.random.choice(ts)
                    random_table = tables.index(rt)
                    ts.pop(ts.index(rt))
                if have_no_attr:
                    break
                """
                cols, ops, vals = GenerateQuery([tables[random_table].columns[available_col_idx]
                                                 for available_col_idx in list(available_attr[random_table])],
                                                rng, tables[random_table], 1,
                                                available_cols=list(available_attr[random_table]),
                                                return_col_idx=True)
                """
                cols, ops, vals = GenerateQuery(tables[random_table].columns,
                                                rng, tables[random_table], 1,
                                                available_cols=list(available_attr[random_table]),
                                                return_col_idx=True)
                node.left_child = Predicate(estimator.tables_dict[random_table], cols[0], ops[0], vals[0])
                left_num += 1
                available_attr[random_table].remove(cols[0])
        # available_node中都是右边没有谓词的节点
        # 从中选取一部分添加谓词
        assert predicates_num_higher_bound - left_num == len(available_node)
        needed_right = rng.randint(0, predicates_num_higher_bound - left_num)
        for j in range(needed_right, -1, -1):
            random_node = rng.choice(available_node, 1)[0]
            assert random_node.right_child is None
            have_no_attr = False
            if len(available_attr[0]) == 1 and len(available_attr[1]) > 1:
                random_table = 1
            elif len(available_attr[1]) == 1 and len(available_attr[0]) > 1:
                random_table = 0
            elif len(available_attr[0]) == 1 and len(available_attr[1]) > 1:
                raise Exception('无可用属性，生成失败')
            else:
                ts = copy.copy(tables)
                random_table = rng.randint(0, len(tables))
                ts.pop(random_table)
                have_no_attr = False
                while len(available_attr[random_table]) <= 1:
                    if len(ts) == 0:
                        have_no_attr = True
                        break
                    rt = np.random.choice(ts)
                    random_table = tables.index(rt)
                    ts.pop(ts.index(rt))
                if have_no_attr:
                    break
            """
            cols, ops, vals = GenerateQuery([tables[random_table].columns[available_col_idx]
                                             for available_col_idx in list(available_attr[random_table])],
                                            rng, tables[random_table], 1,
                                            available_cols=list(available_attr[random_table]),
                                            return_col_idx=True)
            """
            cols, ops, vals = GenerateQuery(tables[random_table].columns,
                                            rng, tables[random_table], 1,
                                            available_cols=list(available_attr[random_table]),
                                            return_col_idx=True)
            fact_cols, fact_ops, fact_vals, fact_dominant_ops = AQP_estimator.ProjectQuery(estimator.fact_tables[random_table], cols, ops, vals)
            estimator.dominant_ops = fact_dominant_ops
            random_node.right_child = Predicate(estimator.tables_dict[random_table], fact_cols, fact_ops, fact_vals)
            available_node.remove(random_node)
            available_attr[random_table].remove(cols[0])

        # 从未选中的列中随机目标列
        target_table = rng.randint(0, len(tables))
        # available_attr[0] = set.intersection(available_attr[0], set([0, 2, 4]))
        # available_attr[1] = set.intersection(available_attr[1], set([3, 4, 5]))
        random_variable = RandomVariable(target_table, rng.choice(list(available_attr[target_table]), 1)[0])
        prob_define = Probability(random_variable, root)
        # 确定内连接方式
        # how = inner_joins[rng.randint(0, 5)]
        how = "="

        def get_val(val):
            if isinstance(val, str):
                left_val = f"'{val}'"
            elif isinstance(val, np.datetime64):
                left_val = f"'{pd.to_datetime(str(val)).strftime('%m/%d/%Y')}'"
            else:
                left_val = val
            return left_val

        def get_pred(pred):
            if pred == '=':
                return '=='
            else:
                return pred

        def dfs(root):
            cond = '&' if root.op_type == 'and' else '|'
            if isinstance(root.left_child, Predicate):
                left_val = get_val(root.left_child.val)
                if isinstance(root.right_child, Predicate):
                    right_val = get_val(root.right_child.val)
                    return f"( `{tables[root.left_child.table].columns[root.left_child.attr].name}` {get_pred(root.left_child.predicate)} {left_val} {cond} `{tables[root.right_child.table].columns[root.right_child.attr].name}` {get_pred(root.right_child.predicate)} {right_val} )"
                elif isinstance(root.right_child, PredicateTreeNode):
                    right = dfs(root.right_child)
                    return f"( `{tables[root.left_child.table].columns[root.left_child.attr].name}` {get_pred(root.left_child.predicate)} {left_val} {cond} " + right + " )"
                else:
                    return f"( `{tables[root.left_child.table].columns[root.left_child.attr].name}` {get_pred(root.left_child.predicate)} {left_val})"
            elif isinstance(root.left_child, PredicateTreeNode):
                left = dfs(root.left_child)
                if isinstance(root.right_child, Predicate):
                    right_val = get_val(root.right_child.val)
                    return "( " + left + f" {cond} `{tables[root.right_child.table].columns[root.right_child.attr].name}` {get_pred(root.right_child.predicate)} {right_val} )"
                elif isinstance(root.right_child, PredicateTreeNode):
                    right = dfs(root.right_child)
                    return "( " + left + f" {cond} " + right + ")"
                else:
                    return left

        query = dfs(prob_define.conditions)
        print(f'query: {query}')
        # if i != 1958:
        #     continue
        if query not in ground_truth_cache['selectivity']:
            table_left = tables[0].data
            table_right = tables[1].data
            if 'Reg Valid Date' in set(table_left.columns):
                table_left['Reg Valid Date'] = pd.to_datetime(table_left['Reg Valid Date'], format='%m/%d/%Y')
            if 'Reg Valid Date' in set(table_right.columns):
                table_right['Reg Valid Date'] = pd.to_datetime(table_right['Reg Valid Date'], format='%m/%d/%Y')
            if table_left.shape[0] > 1e5 or table_right.shape[0] > 1e5:
                table_left = table_left.sample(n=10000)
                table_right = table_right.sample(n=10000)

            estimator.dfs_visit_condition_nodes(prob_define.conditions, prob_define.conditions)
            table_set = set()
            for predicates in prob_define.conditions.predicate_list:
                for predicate in predicates:
                    if predicate.table not in table_set:
                        table_set.add(predicate.table)
            try:
                if len(table_set) == 2:
                    df = pd.merge(table_left, table_right, how='inner', left_on='Color', right_on='Color')
                    joined_df = df.query(query)
                    df_ra = []
                    # print(tables[random_variable.table].columns[random_variable.attr].all_distinct_values)
                    for distinct_val in tables[random_variable.table].columns[random_variable.attr].all_distinct_values:
                        v = get_val(distinct_val)
                        if v is np.nan:
                            v = "Unknown"
                        df_ra.append(
                            len(joined_df.query(
                                f"`{tables[random_variable.table].columns[random_variable.attr].name}` == {v}")))
                    t = len(joined_df)
                    selectivity = joined_df.shape[0] / df.shape[0]
                else:
                    table_num = table_set.pop()
                    selectivity = tables[table_num].data.query(query).shape[0] / tables[table_num].data.shape[0]
            except UndefinedVariableError as e:
                selectivity = 0
            ground_truth_cache['selectivity'][query] = selectivity
        else:
            selectivity = ground_truth_cache['selectivity'][query]
        print(f'selectivity:{selectivity}')
        selectivity_list.append(selectivity)

        st = time.time()
        condition_prob = estimator.get_prob_of_predicate_tree(prob_define.conditions, tables, how,
                                                              num_samples=config['psample'], real=real)
        st = time.time() - st
        print(f'prob:{condition_prob}, time cost: {st} s')
        tc.append(st)
        if selectivity != 0:
            count_RelErr = np.abs(condition_prob - selectivity) / selectivity
            # print(f'selectivity:{selectivity}')
            print(f'count_RelErr:{count_RelErr}')
        else:
            count_RelErr = np.array([0])
        if np.min([condition_prob.item(), selectivity]) != 0:
            q_error = np.max([condition_prob.item(), selectivity]) / np.min([condition_prob.item(), selectivity])
            print(f'q_error:{q_error}')
        else:
            q_error = 1.
        count_error_list.append(count_RelErr)
        q_error_list.append(q_error)

        total_count_RelErr += count_RelErr.item()
        total_q_error += q_error
        with open(f'./Configs/{args.config}/{args.config}_query_ground_truth_seed{seed}.json', 'w',
                  encoding='utf-8') as f:
            json.dump(ground_truth_cache, f)
        # print(total_count_RelErr)

        '''' 
        real_max = joined_df[[tables[random_variable.table].columns[random_variable.attr].name]].max().values[0]
        test_max = test_sample.max().values[0]
        if real_max != 0:
            max_RelErr = np.abs(real_max-test_max) / real_max
            total_max_RelErr += max_RelErr
        else:
            max_RelErr = np.abs(real_max-test_max)
            total_max_num -= 1
        print(f'max_RelErr:{max_RelErr}')
        # print(total_max_RelErr)

        real_min = joined_df[[tables[random_variable.table].columns[random_variable.attr].name]].min().values[0]
        test_min = test_sample.min().values[0]
        if real_min  != 0:
            min_RelErr = np.abs(real_min-test_min) / real_min
            total_min_RelErr += min_RelErr
        else:
            min_RelErr = np.abs(real_min-test_min)
            total_min_num -= 1
        print(f'min_RelErr:{min_RelErr}')
        '''
        # total_min_RelErr += min_RelErr
        # print(total_min_RelErr)
        # prob = estimator.get_prob_with_conditions(Probability(random_variable,prob_define.conditions.left_child), tables, keys, how, num_samples=1000).cpu().numpy()
        # t = joined_df[[tables[random_variable.table].columns[random_variable.attr].name]].sample(n=int(joined_df.shape[0]*prob.mean()))
        # test_avg_1 = t.mean().values[0]
        # test_mae_1 = np.abs(real_avg-test_avg_1) / real_avg
        # total_test_mae_1 += test_mae_1
        # print(f'test_mae_1:{test_mae_1}')
    print(f'final mean mae:{total_mae / num}')
    print(f'final mean mse:{total_mse / num}')
    print(f'final mean cross entropy:{total_cross_entropy / num}')
    print(f'final avg mae:{total_avg_RelErr / total_avg_num}')
    print(f'final sum RelErr:{total_sum_RelErr / total_sum_num}')
    print(f'final max RelErr:{total_max_RelErr / total_max_num}')
    print(f'final min RelErr:{total_min_RelErr / total_min_num}')
    print(f'final count RelErr:{total_count_RelErr / num}')
    print(f'final q error:{total_q_error / num}')
    print(f'mean time cost{np.array(tc).mean()}')
    selectivity_list = np.array(selectivity_list)

    count_error_list = np.array(count_error_list)
    count_error_high = count_error_list[selectivity_list > 0.02]
    count_error_medium = count_error_list[np.bitwise_and(selectivity_list > 0.005, selectivity_list <= 0.02)]
    count_error_low = count_error_list[selectivity_list < 0.005]
    print(f'Count-Error')
    print('\tselectivity high((2%,100%])')
    q_high_df = pd.DataFrame(count_error_high)
    print(
        f'\tMean: {q_high_df.mean().to_numpy().item()}, 50th: {q_high_df.quantile(q=0.5).to_numpy().item()}, 95th: {q_high_df.quantile(q=0.95).to_numpy().item()}, 99th: {q_high_df.quantile(q=0.99).to_numpy().item()}, max:{q_high_df.max().to_numpy().item()}')
    print('\tselectivity medium((0.5%,2%])')
    q_medium_df = pd.DataFrame(count_error_medium)
    print(
        f'\tMean: {q_medium_df.mean().to_numpy().item()}, 50th: {q_medium_df.quantile(q=0.5).to_numpy().item()}, 95th: {q_medium_df.quantile(q=0.95).to_numpy().item()}, 99th: {q_medium_df.quantile(q=0.99).to_numpy().item()}, max:{q_medium_df.max().to_numpy().item()}')
    print('\tselectivity low((0%,0.5%])')
    q_low_df = pd.DataFrame(count_error_low)
    print(
        f'\tMean: {q_low_df.mean().to_numpy().item()}, 50th: {q_low_df.quantile(q=0.5).to_numpy().item()}, 95th: {q_low_df.quantile(q=0.95).to_numpy().item()}, 99th: {q_low_df.quantile(q=0.99).to_numpy().item()}, max:{q_low_df.max().to_numpy().item()}')
    q_df = pd.DataFrame(count_error_list)
    print(
        f'Count-Error: 50th: {q_df.quantile(q=0.5).to_numpy().item()}, 95th: {q_df.quantile(q=0.95).to_numpy().item()}, 99th: {q_df.quantile(q=0.99).to_numpy().item()}, max:{q_df.max().to_numpy().item()}')

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
    print(
        f'Q-Error: 50th: {q_df.quantile(q=0.5).to_numpy().item()}, 95th: {q_df.quantile(q=0.95).to_numpy().item()}, 99th: {q_df.quantile(q=0.99).to_numpy().item()}, max:{q_df.max().to_numpy().item()}')
    # plt.figure(figsize=(10, 10))  # 设置绘图大小为20*15
    plt.yscale('log')
    plt.xlabel('selectivity')  # 设置x、y轴标签
    plt.ylabel('count_error')
    plt.scatter(selectivity_list, count_error_list)
    plt.show()

    # plt.figure(figsize=(10, 10))  # 设置绘图大小为20*15
    plt.yscale('log')
    plt.xlabel('selectivity')  # 设置x、y轴标签
    plt.ylabel('q_error')
    plt.scatter(selectivity_list, q_error_list)
    plt.show()

    # plt.figure(figsize=(10, 10))  # 设置绘图大小为20*15
    # plt.yscale('log')
    plt.ylabel('q_error')
    plt.boxplot(q_error_list, meanline=True, showmeans=True, showfliers=False)
    plt.show()

    # sum = df[tables[random_variable.table].columns[random_variable.attr].name].sum()
    # cond = []
    # for _ in range(3):
    #     table_id = np.random.randint(len(tables))
    #     attr_id = np.random.randint(len(tables[table_id].columns))
    #     op = inner_joins[rng.randint(0, 5)]
    #     val = tables[table_id].columns[attr_id].all_distinct_values[rng.randint(0, len(tables[table_id].columns[attr_id].all_distinct_values)+1)]
    # cond.append((table_id,attr_id,op,val))

    # 计算真实结果
    # query = GenerateQuery(cols, rng, table)
    # Query(estimators,
    #       do_print,
    #       oracle_card=oracle_cards[i]
    #       if oracle_cards is not None and i < len(oracle_cards) else None,
    #       query=query,
    #       table=table,
    #       oracle_est=oracle_est)
    # max_err = ReportEsts(estimators)

    return False


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


def Main():
    models = []
    fact_tables = []
    # Load Tables
    if args.config == 'IMDB':
        tables_name = [name.replace('.csv', '') for name in JoinOrderBenchmark.GetJobLightJoinKeys().keys()]
        tables_dict = MakeIMDBTables(tables_name, use_cols=raw_config['use_cols'])
        tables = list(tables_dict.values())
    else:
        tables = []
        tables_dict = {}
        tables_name = raw_config['dataset']
    # Load Models
    for dataset in tables_name:
        glob = config['glob'].format(dataset, )
        if args.config == 'IMDB':
            table = tables_dict[dataset]
        else:
            table, train_data = MakeTable(dataset)
            tables.append(table)
            tables_dict[dataset] = table
        train_data = common.FactorizedTable(common.TableDataset(table))
        fact_tables.append(train_data)
        z = re.match('.+seed([\d\.]+).*.pt', glob)
        assert z
        seed = int(z.group(1))
        torch.manual_seed(seed)
        for i in range(len(table.columns)):
            print(table.columns[i])
            print(table.columns[i].DistributionSize())
        model = MakeModel(table, train_data)
        ReportModel(model)
        print('Loading ckpt:', glob)
        model.load_state_dict(torch.load(f'./Configs/{args.config}/' + glob))
        model.eval()
        # model.double()
        models.append(model)
        print(glob, seed)
    estimator = DirectEstimator(models,
                                tables_dict,
                                fact_tables,
                                config=config,
                                device=DEVICE)

    RunN(tables,
         estimator,
         # rng=np.random.RandomState(int(time.time())),
         rng=np.random.RandomState(1234),
         num=config['num_queries'],
         log_every=1)

    # SaveEstimators(args.err_csv, estimator)
    # print('...Done, result:', args.err_csv)


if __name__ == '__main__':
    Main()
