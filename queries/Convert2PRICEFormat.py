import os
import pandas as pd

datasets_list = ['imdb']
workload_list = ['job-light-ranges']
dataset_path = '/home/zkx/PRICE/datas/datasets/imdb/'

import psycopg2

def get_cardinality_estimate(query):
    connection = False
    try:
        # 连接到 PostgreSQL 数据库
        connection = psycopg2.connect(
            user="postgres",
            password="123",
            host="127.0.0.1",
            port="55432",
            database="imdb_aqp"
        )

        cursor = connection.cursor()

        # 执行 EXPLAIN 命令
        explain_query = f"EXPLAIN ANALYZE {query}"
        cursor.execute(explain_query)

        # 获取查询计划
        explain_result = cursor.fetchall()

        # 解析查询计划，提取基数估计
        cardinality_estimate = None
        for row in explain_result:
            if 'rows=' in row[0]:
                parts = row[0].split('rows=')
                if len(parts) > 1:
                    cardinality_estimate = int(parts[1].split()[0])
                    break

        return cardinality_estimate

    except Exception as error:
        print(f"Error: {error}")
        return None

    finally:
        if connection:
            cursor.close()
            connection.close()

if __name__ == '__main__':
    for dataset, workload in zip(datasets_list, workload_list):
        with open(f'./{workload}.sql', 'r') as f:
            queries = f.readlines()
        with open(f'./{workload}.csv', 'r') as f:
            csv = f.readlines()
        new_queries = []
        for i, (query, csv_line) in enumerate(zip(queries, csv)):
            if 'series_years' in query or 'phonetic_code' in query or 'season_nr' in query or 'episode_nr' in query or 'nr_order' in query or 'imdb_index' in query:
                continue
            query = query.strip()
            query = query.replace('movie_companies mc', 'movie_companies imdb_mc')
            query = query.replace('movie_info_idx mi_idx', 'movie_info_idx imdb_mii')
            query = query.replace('title t', 'title imdb_t')
            query = query.replace('movie_keyword mk', 'movie_keyword imdb_mk')
            query = query.replace('movie_info mi', 'movie_info imdb_mi')
            query = query.replace('cast_info ci', 'cast_info imdb_ci')

            query = query.replace('mc.', 'imdb_mc.')
            query = query.replace('mi_idx.', 'imdb_mii.')
            query = query.replace('t.', 'imdb_t.')
            query = query.replace('mk.', 'imdb_mk.')
            query = query.replace('mi.', 'imdb_mi.')
            query = query.replace('ci.', 'imdb_ci.')

            card = csv_line.split('#')[-1].strip()
            # pg_card = get_cardinality_estimate(query.replace("COUNT(*)", "*"))
            pg_card = 1
            new_query = '||'.join((query, card, str(pg_card)))
            new_queries.append(new_query)
        import platform


        def get_os_name():
            os_name = platform.system()
            return os_name
        current_os = get_os_name()
        if current_os == "Windows":
            if not os.path.exists('./PRICE/'):
                os.makedirs('./PRICE/')
            path = f'./PRICE/{workload}.sql'
        elif current_os == "Linux":
            path = f'/home/zkx/PRICE/datas/workloads/test/{dataset}/{workload}_1_pg_card.sql'
        with open(path, 'w') as f:
            f.write('\n'.join(new_queries))
