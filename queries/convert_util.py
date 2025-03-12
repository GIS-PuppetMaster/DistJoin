import pandas as pd

def replace_joins(df, how):
    # get join keys
    join_keys_dict = {}
    for join_keys in df.iloc[:, 1]:
        if pd.isna(join_keys):
            continue
        join_keys = join_keys.split(',')
        for jk in join_keys:
            jks = jk.split('=')
            for k in jks:
                l = k.split('.')
                if l[0] not in join_keys_dict.keys():
                    join_keys_dict[l[0]] = l[1]
                else:
                    assert join_keys_dict[l[0]] == l[1]

    replaced_joins = []
    for join_tables_ in df.iloc[:, 0]:
        join_tables = join_tables_.split(',')
        if len(join_tables)==1:
            replaced_joins.append(join_tables_)
            continue
        j_tabs = []
        for jt in join_tables:
            try:
                jt = jt.split(' ')[1]
            except:
                pass # job-light-ranges has no shorten name for each table
            j_tabs.append(jt)
        new_join = []
        for left_jt, right_jt in zip(j_tabs[:-1], j_tabs[1:]):
            new_join.append(f'{left_jt}.{join_keys_dict[left_jt]}{how}{right_jt}.{join_keys_dict[right_jt]}')
        new_join = ','.join(new_join)
        replaced_joins.append(new_join)
    return replaced_joins

    '''              
    join_tabs = []
    for join_tables in df.iloc[:, 0]:
        join_tables = join_tables.split(',')
        j_tabs = []
        for jt in join_tables:
            jt = jt.split(' ')[1]
            j_tabs.append(jt)
        join_tabs.append(j_tabs)
        

    # with the same join order as DistJoin, since non-equi join is sensitive to join order
    def get_join_order(j_tabs):
        left_jt = j_tabs[:len(j_tabs)//2]
        right_jt = j_tabs[len(j_tabs)//2:]
        if len(left_jt)>1:
            order_left = get_join_order(left_jt)
        elif len(left_jt)==1:
            order_left = [left_jt[0]]
        else:
            assert False
        if len(right_jt)>1:
            order_right = get_join_order(right_jt)
        elif len(right_jt)==1:
            order_right = [right_jt[0]]
        else:
            assert False
        return [*order_left, *order_right]
    replaced_joins = []

    for j_tabs in join_tabs:
        new_join = get_join_order(j_tabs)
        replaced_joins.append(new_join)
    return replaced_joins
    '''