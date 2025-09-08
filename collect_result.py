import os
from collections import defaultdict

import pandas as pd
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

import AQP_estimator
from utils.util import get_q_error

train_tags = {'DistJoin': ['', '_previous'],
        'PG':['', '_previous'],
        'NeuroCard':['', '_previous'],
        'FactorJoin':['', '_previous'],
        'MSCN':['', '_previous'],
        'ALECE':['', '_previous'],
        'UAE': ['', '_previous'],
        }
eval_tags = {
    '1740979448.161846': [''],
    # '1740289713.764569': [''],
    '1740983483.941268' : ['', '_previous'],
    # '1740983483.941268': ['', '_previous'],
    'PG': ['', '_previous'],
    'result': ['', '_previous'],
    'FactorJoin': ['', '_previous'],
    'MSCN':['', '_previous'],
    'ALECE': ['', '_previous'],
    'UAE': ['', '_previous'],

}
hows = AQP_estimator.OPS_array.tolist()
methods = ['DistJoin', 'PG', 'NeuroCard', 'FactorJoin', 'MSCN', 'ALECE', 'UAE'] #x must load DistJoin first to get selectivity column
DistJoin_tag_to_exp_marks = {
    'job-light': '1740979448.161846',
    'job-light-ranges': '1740979448.161846',
    # 'job-light_previous':'1739913859.264736',
    'job-light_previous':'1740983483.941268',
    'job-light-ranges_previous': '1740983483.941268',
}
workload_to_num = {
    'job-light':70,
    'job-light-ranges': 1000
}
workload_map_for_FactorJoin = {
    'job-light': 'light',
    'job-light-ranges': 'light_ranges'
}
DistJoin_path = './result/DistJoin'
PG_result_path = './result/PG'
NeuroCard_path = './result/NeuroCard'
FactorJoin_path = './result/FactorJoin'
MSCN_path = './result/MSCN'
ALECE_path = './result/ALECE'
UAE_path = './result/UAE'
result_dict = {'method': [], 'train_tag':[], 'eval_tag':[], 'how':[], 'workload':[], 'q_error':[], 'rel_error':[], 'cost':[], 'selectivity':[], 'est_card':[], 'true_card':[]}

def get_exp_mark(method):
    if method == 'DistJoin':
        exp_mark = DistJoin_tag_to_exp_marks[workload + train_tag]
    elif method == 'PG':
        exp_mark = 'PG'
    elif method in ['NeuroCard', 'UAE']:
        exp_mark = 'result'
    else:
        exp_mark = method
    return exp_mark

for method in methods:
    for workload in ['job-light', 'job-light-ranges']:
        for train_tag in train_tags[method]:
            exp_mark = get_exp_mark(method)
            for eval_tag in eval_tags[exp_mark]:
                if train_tag == '' and eval_tag == '_previous':
                    continue
                if method == 'PG' and train_tag!=eval_tag:
                    continue
                for how in hows:
                    distjoin_mark = get_exp_mark('DistJoin')
                    distjoin_df = pd.read_csv(os.path.join(DistJoin_path, distjoin_mark, f'{distjoin_mark}_{workload}_{how}{eval_tag}.csv'))
                    if method == 'DistJoin':
                        df = distjoin_df
                    elif method == 'PG':
                        df = pd.read_csv(os.path.join(PG_result_path, f'{workload}_{how}{eval_tag}.csv'))
                    elif method == 'NeuroCard':
                        if how != '=':
                            continue
                        df = pd.read_csv(os.path.join(NeuroCard_path, f'{exp_mark}_[\'{workload}{train_tag}{eval_tag}\'].csv'))
                    elif method == 'FactorJoin':
                        if how != '=':
                            continue
                        df = pd.read_csv(os.path.join(FactorJoin_path, f'{workload_map_for_FactorJoin[workload]}{train_tag}{eval_tag}.csv'))
                    elif method == 'MSCN':
                        df = pd.read_csv(os.path.join(MSCN_path, f'{workload}-{workload_to_num[workload]}queries-seed42-{how}{train_tag}{eval_tag}.csv'))
                    elif method == 'ALECE':
                        df = pd.read_csv(os.path.join(ALECE_path, f'{workload}-{how}{eval_tag}.csv'))
                    elif method == 'UAE':
                        if how != '=':
                            continue
                        df = pd.read_csv(os.path.join(UAE_path, f'{exp_mark}_[\'{workload}-mscn-workload{train_tag.replace('_', '-')}{eval_tag.replace('_', '-')}\'].csv'))
                    else:
                        raise Exception(f'not supported method:{method}')
                    if method in ['FactorJoin', 'ALECE']: # fix the outdated true_card and errors
                        df['true_card'] = distjoin_df['true_card'] if len(df) == len(distjoin_df) else distjoin_df['true_card'][(distjoin_df['true_card']!=0) & (distjoin_df['true_card']!='0')]
                        est_cards = df['est_card']
                        true_cards = np.array([int(c) for c in list(df['true_card'])],dtype=object)
                        df['q_error'] = np.array([get_q_error(int(est_card), int(true_card)) for est_card, true_card in zip(est_cards, true_cards)])
                        df['rel_error'] = np.array([np.abs(est_card - true_card) / true_card if true_card!=0 else np.nan for est_card, true_card in zip(est_cards, true_cards)])
                    df = df[(df['true_card'] != 0) & (df['true_card'] != '0')]
                    print(f'{method} {workload}, {train_tag}, {eval_tag}, {how}')
                    if 'selectivity' in df.columns:
                        result_dict['selectivity'].extend(df['selectivity'])
                    else:
                        tmp = pd.DataFrame(result_dict)
                        tmp = tmp[(tmp['method'] == 'DistJoin') & (tmp['train_tag'] == train_tag) & (tmp['eval_tag'] == eval_tag) & (tmp['how'] == how) & (tmp['workload'] == workload)]
                        sel = tmp['selectivity'].values.tolist()
                        assert len(sel)==df.shape[0]
                        # make sure the experiment is consisting
                        # assert (df['true_card'] == tmp['true_card']).all()
                        result_dict['selectivity'].extend(sel)

                    result_dict['workload'].extend([workload for _ in range(df.shape[0])])
                    result_dict['q_error'].extend(df['q_error'])
                    result_dict['rel_error'].extend(df['rel_error'])
                    result_dict['cost'].extend(df['cost']*1000) # s to ms
                    result_dict['method'].extend([method for _ in range(df.shape[0])])
                    result_dict['train_tag'].extend([train_tag for _ in range(df.shape[0])])
                    result_dict['eval_tag'].extend([eval_tag for _ in range(df.shape[0])])
                    result_dict['how'].extend([how for _ in range(df.shape[0])])
                    result_dict['est_card'].extend(df['est_card'])
                    result_dict['true_card'].extend([int(v) for v in df['true_card'].values])
result_df = pd.DataFrame(result_dict)
result_df.to_csv('result.csv', index=False, na_rep="")
result_summary = {'method': [], 'train_tag':[], 'eval_tag':[], 'how':[], 'workload':[], 'q_error_mean':[],'q_error_50th':[],'q_error_95th':[],'q_error_99th':[],'q_error_100th':[], 'rel_error_mean':[], 'cost_mean':[]}
# calculate percentile error
result_df = result_df[(result_df['true_card']!=0) & (result_df['true_card']!='0')]
for method in methods:
    for workload in ['job-light', 'job-light-ranges']:
        for train_tag in train_tags[method]:
            exp_mark = get_exp_mark(method)
            for eval_tag in eval_tags[exp_mark]:
                if train_tag == '' and eval_tag == '_previous':
                    continue
                if method == 'PG' and train_tag!=eval_tag:
                    continue
                for how in hows:
                    if method in ['NeuroCard', 'FactorJoin', 'UAE'] and how != '=':
                        continue
                    filted = result_df[(result_df['method'] == method) & (result_df['train_tag'] == train_tag) & (result_df['eval_tag'] == eval_tag) & (result_df['how'] == how) & (result_df['workload'] == workload)]
                    result_summary['method'].append(method)
                    result_summary['train_tag'].append(train_tag)
                    result_summary['eval_tag'].append(eval_tag)
                    result_summary['how'].append(how)
                    result_summary['workload'].append(workload)
                    result_summary['q_error_mean'].append(filted['q_error'].mean())
                    result_summary['q_error_50th'].append(filted['q_error'].quantile(0.5).item())
                    result_summary['q_error_95th'].append(filted['q_error'].quantile(0.95).item())
                    result_summary['q_error_99th'].append(filted['q_error'].quantile(0.99).item())
                    result_summary['q_error_100th'].append(filted['q_error'].max())
                    result_summary['rel_error_mean'].append(filted['rel_error'].mean())
                    result_summary['cost_mean'].append(filted['cost'].mean())
result_summary_df = pd.DataFrame(result_summary)
result_summary_df = result_summary_df.round(2)
result_summary_df.to_csv('result_summary.csv', index=False)


summary = defaultdict(list)
total = None
for workload in ['job-light', 'job-light-ranges']:
    exp_mark = DistJoin_tag_to_exp_marks[workload]

    all_table_costs = None
    files = os.listdir(f'./train_log/{exp_mark}/')
    epoch_num = 20
    for file in files:
        if '.yaml' in file:
            continue
        table_name = file.replace('.csv', '')
        df = pd.read_csv(f'./train_log/{exp_mark}/{file}')
        df['table'] = table_name

        max_epoch = df['epoch'].max()
        if max_epoch < epoch_num - 1:
            last_row = df[df['epoch'] == max_epoch].iloc[-1].copy()

            # 生成需要补充的epoch
            new_epochs = range(max_epoch + 1, 20)

            # 创建新数据行
            new_rows = [last_row.copy() for _ in new_epochs]
            for i, e in enumerate(new_epochs):
                new_rows[i]['epoch'] = e

            # 合并数据
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        all_table_costs = df if all_table_costs is None else pd.concat([all_table_costs, df])


    path = f'./result/DistJoin/{exp_mark}/{workload}_by_epoch.csv'
    df = pd.read_csv(path)
    epochs = np.unique(df['epoch'].values)
    times = []
    for epoch in epochs:
        df_epoch = df[df['epoch']==epoch]
        summary['epoch'].append(epoch)
        summary['workload'].append(workload)
        summary['q_error_mean'].append(df_epoch['q_error'].mean())
        summary['q_error_50th'].append(df_epoch['q_error'].quantile(0.5).item())
        summary['q_error_95th'].append(df_epoch['q_error'].quantile(0.95).item())
        summary['q_error_99th'].append(df_epoch['q_error'].quantile(0.99).item())
        summary['q_error_100th'].append(df_epoch['q_error'].max())
        time = all_table_costs[all_table_costs['epoch']==epoch]['seconds_since_start'].max()
        summary['time'].append(time)
        times.extend([time]*len(df_epoch))
    df['time'] = times
    df['workload'] = workload
    total = df if total is None else pd.concat([total, df])
total.to_csv('total_by_epoch.csv', index=False)
pd.DataFrame(summary).to_csv('summary_by_epoch.csv', index=False)