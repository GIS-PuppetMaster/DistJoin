import psycopg2
from psycopg2 import sql
import psycopg2.extras
import pandas as pd
import os
from tqdm import tqdm

from datasets import JoinOrderBenchmark

connection = False
workload_path = '../../queries/'
workloads = ('job-light', 'job-light-ranges')
data_path = './'
tag = '_previous'
def create_schema(connection, create_sqls):
    cursor = connection.cursor()
    for sql in create_sqls:
        if sql != '':
            cursor.execute(sql)
    connection.commit()
    cursor.close()
def upload_csv(connection, name, df):
    cursor = connection.cursor()
    table = name.replace('.csv','')
    # 将数据插入到新的表中

    if len(df) > 0:
        df_columns = list(df)
        # create (col1,col2,...)
        columns = ",".join(df_columns)

        # create VALUES('%s', '%s",...) one '%s' per column
        values = "VALUES({})".format(",".join(["%s" for _ in df_columns]))

        # create INSERT INTO table (columns) VALUES('%s',...)
        insert_stmt = "INSERT INTO {} ({}) {}".format(table, columns, values)
        # psycopg2.extras.execute_batch(cursor, insert_stmt, df.values.astype(object))
        vals = df.values.astype(object)
        vals[pd.isnull(vals)] = None
        if table == 'movie_companies':
            for row in vals:
                cursor.execute(insert_stmt, row)
            connection.commit()
        else:
            for s, e in zip(list(range(0, len(vals), 10000)), list(range(0, len(vals), 10000))[1:]+[len(vals)]):
                psycopg2.extras.execute_batch(cursor, insert_stmt, vals[s:e])
                connection.commit()
            # psycopg2.extras.execute_batch(cursor, insert_stmt, vals)
        # connection.commit()
        cursor.close()

    # insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({});").format(
    #     sql.Identifier(name.replace('.csv','')),  # 表名
    #     sql.SQL(', ').join(map(sql.Identifier, df.columns)),  # 列名
    #     sql.SQL(', ').join(map(sql.Placeholder, df.columns))  # 占位符
    # )
    # data_to_insert = [tuple(row) for row in df.itertuples(index=False, name=None)]
    # cursor.executemany(str(insert_query), data_to_insert)
    # for row in df.itertuples(index=False, name=None):
    #     cursor.execute(insert_query, row)

    # 提交事务
    connection.commit()
    cursor.close()

# try:
# 连接到 PostgreSQL 数据库
connection = psycopg2.connect(
    user="postgres",
    password="123",
    host="127.0.0.1",
    port="5432",
    database="aqp_previous"
)
connection.set_client_encoding('UTF8')
cursor = connection.cursor()

# 导入csv
csv_map = {}
create_schema_sql = None
tables_name = [name+tag for name in JoinOrderBenchmark.GetJobLightJoinKeys().keys()]
# for file in os.listdir(data_path):
#     if '.csv' in file and file.replace('.csv','') in tables_name:
#         print(f'load data:{file}')
#         csv_map[file.split('\\.')[0]] = pd.read_csv(os.path.join(data_path, file),escapechar='\\', low_memory=False)
#     elif 'schematext.sql' in file:
#         print('load schema')
#         with open(os.path.join(data_path, file), 'r') as f:
#             create_schema_sql = f.readlines()

# try:
#     create_schema(connection, "".join(create_schema_sql).strip().replace("\n","").split(";"))
# except Exception as e:
#     print(e)
# for name, df in csv_map.items():
#     upload_csv(connection, name.replace(tag, ''), df)

for workload in workloads:
    with open(os.path.join(workload_path, workload+'.csv'), 'r') as f:
        workload_csv = f.readlines()
    with open(os.path.join(workload_path, workload+'.sql'), 'r') as f:
        sqls = f.readlines()
    assert len(workload_csv) == len(sqls)
    for i, query_sql in enumerate(tqdm(sqls, desc=workload)):
        cursor = connection.cursor()
        # 执行查询
        cursor.execute(query_sql)
        # 获取结果
        records = cursor.fetchall()
        workload_csv_line = workload_csv[i].split('#')
        workload_csv_line[-1] = str(records[0][0]) + '\n'
        workload_csv_line = '#'.join(workload_csv_line)
    with open(os.path.join(workload_path, workload + '_my.csv'), 'w') as f:
        f.writelines(workload_csv)
if connection:
    cursor.close()
    connection.close()
# except (Exception, psycopg2.Error) as error:
#     print("发生错误，错误信息：", error)

# finally:
#     # 关闭数据库连接
#     if connection:
#         cursor.close()
#         connection.close()