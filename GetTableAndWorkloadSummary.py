import numpy as np
import pandas as pd
import utils
from datasets import JoinOrderBenchmark
from utils.util import MakeIMDBTables
import pickle as pkl
import matplotlib.pyplot as plt
cmap = plt.get_cmap("Set2")
styles = {
    'job-light':  'solid',
    'job-light-ranges': '--',
    '=': cmap(0),
    '>': cmap(1),
    '<': cmap(2),
    '>=': cmap(3),
    '<=': cmap(4),
}
map_how = {
    '=':'$=$',
    '>':'$>$',
    '<':'$<$',
    '>=':'$\geq$',
    '<=':'$\leq$'
}
workloads = ['job-light',
             'job-light-ranges']
bin_nums = {'job-light': 5,
            'job-light-ranges': 50}
hows = ['=', '>', '<', '>=', '<=']
def draw_curve(true_cards, base_cards):
    plt.figure(figsize=(8, 2.5), dpi=300)
    for workload_name in workloads:
        for how in hows:
            true_card = true_cards[workload_name+how]
            base_card = base_cards[workload_name+how]
            selectivity = np.sort(true_card/base_card)
            bins = pd.qcut(selectivity, bin_nums[workload_name], duplicates='drop')
            x = np.array(list(map(lambda x: x.right, bins.categories)))
            # x = np.insert(x, 0, 0.)
            counts = np.cumsum(np.unique(bins.codes, return_counts=True)[1] / len(true_card))
            # counts = np.insert(counts, 0, 0.)
            assert x.shape == counts.shape
            # plt.step(x, counts, where='post', color=styles[dataset][0], linestyle=styles[dataset][1])
            plt.plot(x, counts, color=styles[how], linestyle=styles[workload_name],
                     label=f'{workload_name.replace('job','JOB')}, Join Condition: {map_how[how]}')
    plt.xscale('log')
    plt.legend(prop={'size':6})
    plt.xlabel('Query Selectivity (log scale)')
    plt.ylabel('Percentile of query')
    plt.tight_layout()
    plt.grid()
    import os
    os.makedirs('./result/figs/', exist_ok=True)
    plt.savefig('./result/figs/workload_dist.png',bbox_inches='tight')
    plt.show()

# 假设 df 是你的 DataFrame
def summarize_table(df):
    # 行数和列数
    num_rows, num_cols = df.shape

    # 列类型（分类/连续数值）
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):  # 判断是否为数值类型
            column_types[col] = 'Continuous'
        else:
            column_types[col] = 'Categorical'

    # 列的 distinct value 数
    distinct_values = {col: df[col].nunique() for col in df.columns}

    # 汇总信息
    summary = {
        'Number of Rows': num_rows,
        'Number of Columns': num_cols,
        'Column Types': column_types,
        'Distinct Values per Column': distinct_values
    }

    return summary

JoinOrderBenchmark.LoadTrueBaseCard('')
tables_dict={}
for table_name, key in JoinOrderBenchmark.GetJobLightJoinKeys().items():
    print(f'table name:{table_name}')
    table = MakeIMDBTables(table_name, data_dir='./datasets/job/', use_cols=None)
    tables_dict[table_name] = table
    summary = summarize_table(table.data)
    print(summary)


all_true_cards = {}
all_base_cards = {}
for workload in workloads:
    queries_job_format, _ = utils.util.JobToQuery(f'./queries/{workload}.csv')
    loaded_queries = utils.util.UnpackQueries(tables_dict, queries_job_format)
    preds = []
    for query in loaded_queries:
        num = 0
        for p in query[2].values():
            num+=len(p['vals'])
        preds.append(num)
    preds = np.array(preds)
    assert len(preds) == len(loaded_queries)
    print(f'workload: {workload}, min_preds_num:{np.format_float_scientific(preds.min())}, max_preds_num:{np.format_float_scientific(preds.max())}')
    for how in hows:
        workload_tag = workload.split('/')[-1].split('.')[0]
        # windows doesn't support '>=' in file name, use AQP_estimator.OPS_dict[how] instead of how
        true_card_file = f'./queries/{workload_tag}_{how}{''}.pkl'
        with open(true_card_file, 'rb') as f:
            true_card_list = np.array(pkl.load(f))
        base_cards = []
        for query_id, (join_tables, join_keys, preds, true_card) in enumerate(loaded_queries):
            base_cards.append(JoinOrderBenchmark.TRUE_JOIN_BASE_CARDINALITY[how][str(join_tables)])
        base_cards = np.array(base_cards)
        print(f'workload: {workload_tag}, how: {how}, min_card:{np.format_float_scientific(true_card_list.min())}, max_card:{np.format_float_scientific(true_card_list.max())}, min_base_card:{np.format_float_scientific(base_cards.min())}, max_base_card:{np.format_float_scientific(base_cards.max())}')
        all_true_cards[workload_tag+how] = true_card_list
        all_base_cards[workload_tag+how] = base_cards
draw_curve(all_true_cards, all_base_cards)