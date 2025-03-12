import sqlglot
from sqlglot import parse_one, expressions
def convert_sql_to_string(sql):
    # 解析 SQL 查询
    parsed = parse_one(sql)

    # 获取表名和别名
    tables = []
    for table in parsed.find_all(expressions.Table):
        if table.alias:
            tables.append(f"{table.name} {table.alias}")
        else:
            tables.append(table.name)

    joins = []
    conditions = []

    def extract_conditions(expression):
        if isinstance(expression, expressions.And):
            extract_conditions(expression.left)
            extract_conditions(expression.right)
        elif isinstance(expression, expressions.EQ):
            left = expression.left.sql()
            right = expression.right.sql()
            if '.' in left and '.' in right:
                joins.append(f"{left}={right}")
            else:
                conditions.append(f"{left},=,{right}")
        elif isinstance(expression, expressions.GTE):
            left = expression.left.sql()
            right = expression.right.sql()
            conditions.append(f"{left},>=,{right}")
        elif isinstance(expression, expressions.LTE):
            left = expression.left.sql()
            right = expression.right.sql()
            conditions.append(f"{left},<=,{right}")
        elif isinstance(expression, expressions.GT):
            left = expression.left.sql()
            right = expression.right.sql()
            conditions.append(f"{left},>,{right}")
        elif isinstance(expression, expressions.LT):
            left = expression.left.sql()
            right = expression.right.sql()
            conditions.append(f"{left},<,{right}")
        elif isinstance(expression, expressions.NE):
            left = expression.left.sql()
            right = expression.right.sql()
            conditions.append(f"{left},<>,{right}")
        elif isinstance(expression, expressions.Condition):
            left = expression.left.sql()
            right = expression.right.sql()
            conditions.append(f"{left},{expression.token_type},{right}")

    # 处理 WHERE 子句
    where = parsed.args.get("where")
    if where:
        extract_conditions(where.this)

    # 拼接最终字符串
    result = "#".join([
        ",".join(tables),
        ",".join(joins),
        ",".join(conditions)
    ])

    return result

if __name__ == '__main__':
    datasets_list = ['imdb']
    workload_list = ['job-light-ranges']
    for dataset, workload in zip(datasets_list, workload_list):
        with open(f'./PRICE/{workload}.sql', 'r') as f:
            sqls = f.readlines()
        new_csv = []
        for query in sqls:
            query = query.strip()
            query = query.replace('movie_companies imdb_mc', 'movie_companies mc')
            query = query.replace('movie_info_idx imdb_mii', 'movie_info_idx mi_idx')
            query = query.replace('title imdb_t', 'title t')
            query = query.replace('movie_keyword imdb_mk', 'movie_keyword mk')
            query = query.replace('movie_info imdb_mi', 'movie_info mi')
            query = query.replace('cast_info imdb_ci', 'cast_info ci')

            query = query.replace('imdb_mc.', 'mc.')
            query = query.replace('imdb_mii.', 'mi_idx.')
            query = query.replace('imdb_t.', 't.')
            query = query.replace('imdb_mk.', 'mk.')
            query = query.replace('imdb_mi.', 'mi.')
            query = query.replace('imdb_ci.', 'ci.')
            sql, true_card, pg_card = query.split("||")
            new_csv.append(convert_sql_to_string(sql) + "#" + true_card)
        with open(f'./PRICE/{workload}.csv', 'w') as f:
            f.write("\n".join(new_csv))

