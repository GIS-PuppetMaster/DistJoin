import csv
import os
import re
import time

import psycopg2
import pandas as pd
import pickle as pkl
from queries.convert_util import *
import AQP_estimator

hows = AQP_estimator.OPS_array.tolist()
test_workloads = ['job-light', 'job-light-ranges']
shorten_table_name = {
    'title': 't',
    'cast_info': 'ci',
    'movie_info': 'mi',
    'movie_keyword': 'mk',
    'movie_companies': 'mc',
    'movie_info_idx': 'mi_idx',
}
tags = ['', '_previous']

# 数据库连接参数
db_config = dict(
    user="postgres",
    password="123",
    host="127.0.0.1",
    port="5432",
    database="aqp_previous"
)

def auto_quote(value):
    # 如果是数字或能转换为数字的字符串，则不加引号
    try:
        float(value)  # 检查是否可以转换为整数
        return value
    except ValueError:
        # 否则加引号处理字符串
        return f"'{value}'"


for tag in tags:
    db_config['database'] = 'imdb'+tag+'_raw'
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    query_template = "EXPLAIN (ANALYZE OFF) SELECT * FROM {} WHERE {} AND {};"
    for workload in test_workloads:
        for how in hows:
            with open(f'./queries/{workload}_{how}{tag}.pkl', 'rb') as f:
                true_cards = pkl.load(f)
            df = pd.read_csv(f'./queries/{workload}.csv', delimiter='#', escapechar='\\', header=None, quotechar='"')
            card_result = []
            q_errors = []
            rel_errors= []
            costs = []
            replaced_joins = replace_joins(df, how)
            for i in range(len(df)):
                tables = df.iloc[i, 0]
                # joins = df.iloc[i, 1]
                preds = df.iloc[i, 2]
                # replace join how
                joins = replaced_joins[i]
                # process preds
                preds = preds.split(',')
                print(f'query_id: {i}, tables: {tables}, joins: {joins}, preds: {preds}')
                new_preds = []
                for j in range(0, len(preds),3):
                    new_preds.append(f'{preds[j]}{preds[j+1]}{auto_quote(preds[j+2])}')
                query = query_template.format(tables, joins.replace(',', ' AND '), ' AND '.join(new_preds))
                st = time.time()
                cursor.execute(query)
                costs.append(time.time()-st)
                result = cursor.fetchall()
                got_match = False
                for row in result:
                    # 使用正则表达式查找基数估算值
                    match = re.search(r'rows=(\d+)', row[0])  # row[0] 包含查询计划的字符串
                    if match:
                        got_match = True
                        card_result.append(int(match.group(1)))
                        q_errors.append(max(card_result[-1]/true_cards[i], true_cards[i]/card_result[-1]) if true_cards[i] != 0 else float('inf'))
                        rel_errors.append(abs(card_result[-1]-true_cards[i])/true_cards[i] if true_cards[i] != 0 else float('inf'))
                        break
                if not got_match:
                    raise Exception(f'didn\'t get match on {workload}_{how}{tag}.csv line {i}')
            os.makedirs('./result/PG/', exist_ok=True)
            df_res = {'true_card': true_cards, 'est_card': card_result, 'q_error': q_errors, 'rel_error': rel_errors, 'cost': costs}
            df_res = pd.DataFrame(df_res).to_csv(f'./result/PG/{workload}_{how}{tag}.csv', index=False)
    # 关闭连接
    cursor.close()
    conn.close()
