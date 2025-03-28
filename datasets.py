"""Dataset registrations."""
import os
import warnings

import numpy as np
import pandas as pd

import common
import utils.util

"""Registry of datasets and schemas."""
import collections
import os
import pickle

import numpy as np
import pandas as pd


def CachedReadCsv(filepath, **kwargs):
    """Wrapper around pd.read_csv(); accepts same arguments."""
    parsed_path = filepath[:-4] + '.df'
    if os.path.exists(parsed_path):
        with open(parsed_path, 'rb') as f:
            df = pickle.load(f)
        assert isinstance(df, pd.DataFrame), type(df)
        print('Loaded parsed csv from', parsed_path)
    else:
        df = pd.read_csv(filepath, **kwargs)
        with open(parsed_path, 'wb') as f:
            # Use protocol=4 since we expect df >= 4GB.
            pickle.dump(df, f, protocol=4)
        print('Saved parsed csv to', parsed_path)
    return df


class JoinOrderBenchmark(object):
    ALIAS_TO_TABLE_NAME = {
        'ci': 'cast_info',
        'ct': 'company_type',
        'mc': 'movie_companies',
        't': 'title',
        'cn': 'company_name',
        'k': 'keyword',
        'mi_idx': 'movie_info_idx',
        'it': 'info_type',
        'mi': 'movie_info',
        'mk': 'movie_keyword',
    }

    # Columns where only equality filters make sense.
    CATEGORICAL_COLUMNS = collections.defaultdict(
        list,
        {
            # 216
            'company_name': ['country_code'],
            'keyword': [],
            # 113
            'info_type': ['info'],
            # 4
            'company_type': ['kind'],
            # 2
            'movie_companies': ['company_type_id'],
            # Dom size 134K.
            'movie_keyword': ['keyword_id'],
            # 7
            'title': ['kind_id'],
            # 5
            'movie_info_idx': ['info_type_id'],
            # 11
            'cast_info': ['role_id'],
            # 71
            'movie_info': ['info_type_id'],
        })

    # Columns with a reasonable range/IN interpretation.
    RANGE_COLUMNS = collections.defaultdict(
        list,
        {
            # 18487, 17447
            'company_name': ['name_pcode_cf', 'name_pcode_sf'],
            # 15482
            'keyword': ['phonetic_code'],
            'info_type': [],
            'company_type': [],
            'movie_companies': [],
            'movie_keyword': [],
            # 26, 133, 23260, 97, 14907, 1409
            'title': [
                'imdb_index', 'production_year', 'phonetic_code', 'season_nr',
                'episode_nr', 'series_years'
            ],
            'movie_info_idx': [],
            # 1095
            'cast_info': ['nr_order'],
            'movie_info': [],
        })

    CSV_FILES = [
        'name.csv', 'movie_companies.csv', 'aka_name.csv', 'movie_info.csv',
        'movie_keyword.csv', 'person_info.csv', 'comp_cast_type.csv',
        'complete_cast.csv', 'char_name.csv', 'movie_link.csv',
        'company_type.csv', 'cast_info.csv', 'info_type.csv',
        'company_name.csv', 'aka_title.csv', 'kind_type.csv', 'role_type.csv',
        'movie_info_idx.csv', 'keyword.csv', 'link_type.csv', 'title.csv'
    ]

    BASE_TABLE_PRED_COLS = collections.defaultdict(
        list,
        {
            'movie_info_idx.csv': ['info_type_id', 'movie_id'],
            'movie_companies.csv': [
                'company_id', 'company_type_id', 'movie_id'
            ],
            # Column(kind_id, distribution_size=7), Column(production_year,
            # distribution_size=133),
            'title.csv': ['id', 'kind_id', 'production_year'],
            'cast_info.csv': ['movie_id', 'role_id'],
            # Column(keyword_id, distribution_size=134170)
            'movie_keyword.csv': ['movie_id', 'keyword_id'],
            # Column(info_type_id, distribution_size=71), Column(info,
            # distribution_size=2720930), Column(note, distribution_size=133604
            'movie_info.csv': [
                'movie_id',
                'info_type_id',
            ],
            'comp_cast_type.csv': ['id', 'kind'],
            'aka_name.csv': ['id', 'person_id'],
            'name.csv': ['id'],
        })

    JOB_M_PRED_COLS = collections.defaultdict(
        list, {
            'title.csv': [
                'id', 'kind_id', 'title', 'production_year', 'episode_nr'
            ],
            'aka_title.csv': ['movie_id'],
            'cast_info.csv': ['movie_id', 'note'],
            'complete_cast.csv': ['subject_id', 'movie_id'],
            'movie_companies.csv': [
                'company_id', 'company_type_id', 'movie_id', 'note'
            ],
            'movie_info.csv': ['movie_id', 'info', 'note'],
            'movie_info_idx.csv': ['info_type_id', 'movie_id', 'info'],
            'movie_keyword.csv': ['keyword_id', 'movie_id'],
            'movie_link.csv': ['link_type_id', 'movie_id'],
            'kind_type.csv': ['id', 'kind'],
            'comp_cast_type.csv': ['id', 'kind'],
            'company_name.csv': ['id', 'country_code', 'name'],
            'company_type.csv': ['id', 'kind'],
            'info_type.csv': ['id', 'info'],
            'keyword.csv': ['id', 'keyword'],
            'link_type.csv': ['id', 'link'],
        })

    JOB_FULL_PRED_COLS = collections.defaultdict(
        list, {
            'title.csv': [
                'id', 'kind_id', 'title', 'production_year', 'episode_nr'
            ],
            'aka_name.csv': ['person_id'],
            'aka_title.csv': ['movie_id'],
            'cast_info.csv': [
                'person_id', 'person_role_id', 'role_id', 'movie_id', 'note'
            ],
            'char_name.csv': ['id'],
            'comp_cast_type.csv': ['id', 'kind'],
            'comp_cast_type__complete_cast__status_id.csv': ['id', 'kind'],
            'comp_cast_type__complete_cast__subject_id.csv': ['id', 'kind'],
            'company_name.csv': ['id', 'country_code', 'name'],
            'company_type.csv': ['id', 'kind'],
            'complete_cast': ['status_id', 'subject_id', 'movie_id'],
            'info_type.csv': ['id', 'info'],
            'info_type__movie_info__info_type_id.csv': ['id', 'info'],
            'info_type__movie_info_idx__info_type_id.csv': ['id', 'info'],
            'info_type__person_info__info_type_id.csv': ['id', 'info'],
            'keyword.csv': ['id', 'keyword'],
            'kind_type.csv': ['id', 'kind'],
            'link_type.csv': ['id', 'link'],
            'movie_companies.csv': [
                'company_id', 'company_type_id', 'movie_id', 'note'
            ],
            'movie_info_idx.csv': ['info_type_id', 'movie_id', 'info'],
            'movie_info.csv': ['info_type_id', 'movie_id', 'info', 'note'],
            'movie_keyword.csv': ['keyword_id', 'movie_id'],
            'movie_link.csv': ['link_type_id', 'movie_id', 'linked_movie_id'],
            'name.csv': ['id'],
            'person_info.csv': ['person_id', 'info_type_id'],
            'role_type.csv': ['id'],
        })

    # For JOB-light schema.
    TRUE_FULL_OUTER_CARDINALITY = {
        ('cast_info', 'movie_keyword', 'title'): 241319266,
        ('cast_info', 'movie_companies', 'movie_info',\
         'movie_info_idx', 'movie_keyword', 'title'): 2128877229383,
        ('aka_title', 'cast_info', 'comp_cast_type', 'company_name',\
         'company_type', 'complete_cast', 'info_type', 'keyword', \
         'kind_type', 'link_type', 'movie_companies', 'movie_info', \
         'movie_info_idx', 'movie_keyword', 'movie_link', 'title'): 11244784701309,
        ('aka_name', 'aka_title', 'cast_info', 'char_name',\
         'comp_cast_type__complete_cast__status_id', 'comp_cast_type__complete_cast__subject_id',\
         'company_name', 'company_type', 'complete_cast', 'info_type__movie_info__info_type_id', \
         'info_type__movie_info_idx__info_type_id', 'info_type__person_info__info_type_id', 'keyword',\
         'kind_type', 'link_type', 'movie_companies', 'movie_info', 'movie_info_idx', 'movie_keyword',\
         'movie_link', 'name', 'person_info', 'role_type', 'title'): 282014040554480
    }
    TRUE_JOIN_BASE_CARDINALITY = {}
    @ staticmethod
    def LoadTrueBaseCard(tag):
        _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'queries/basecard{tag}.pkl')
        if os.path.exists(_path):
            with open(_path, 'rb') as f:
                JoinOrderBenchmark.TRUE_JOIN_BASE_CARDINALITY = pickle.load(f)
        else:
            warnings.warn(f'can\'t find basecard in {_path}, this is a serious error unless you are generating it or using \'real\' mode for debug')
    '''
    {
        frozenset(('cast_info', 'title', 'movie_keyword', 'movie_companies')): 3101350543,
        frozenset(('title', 'movie_info', 'movie_companies', 'movie_keyword')): 4847757474,
        frozenset(('movie_companies', 'title', 'movie_keyword')): 34863638,
        frozenset(('title', 'movie_info', 'movie_info_idx', 'movie_keyword')): 696176855,
        frozenset(('cast_info', 'title', 'movie_companies')): 80274241,
        frozenset(('title', 'movie_keyword', 'movie_companies', 'movie_info')): 4847757474,
        frozenset(('title', 'movie_info', 'movie_info_idx', 'movie_companies')): 177388547,
        frozenset(('title', 'movie_info', 'movie_keyword')): 235420417,
        frozenset(('movie_keyword', 'title', 'cast_info')): 215765477,
        frozenset(('title', 'movie_info', 'movie_companies')): 67371414,
        frozenset(('title', 'movie_info', 'movie_companies', 'cast_info')): 5783682388,
        frozenset(('title', 'movie_info', 'movie_info_idx', 'cast_info')): 1119077963,
        frozenset(('title', 'movie_info', 'movie_info_idx', 'movie_keyword', 'movie_companies')): 15099690486,
        frozenset(('title', 'movie_keyword', 'movie_companies')): 34863638,
        frozenset(('title', 'cast_info', 'movie_keyword', 'movie_info_idx')): 607273961,
        frozenset(('title', 'cast_info', 'movie_keyword')): 215765477,
        frozenset(('title', 'movie_info_idx', 'movie_keyword')): 10424626,
        frozenset(('cast_info', 'title')): 36244344,
        frozenset(('title', 'movie_info', 'movie_companies', 'cast_info', 'movie_keyword')): 674076224290,
        frozenset(('title', 'cast_info', 'movie_companies')): 80274241,
        frozenset(('title', 'movie_info', 'movie_info_idx', 'cast_info', 'movie_keyword')): 78771931233,
        frozenset(('title', 'movie_info', 'movie_companies', 'movie_info_idx')): 177388547,
        frozenset(('movie_companies', 'title', 'movie_info_idx')): 4073078,
    }
    '''

    # CSV -> RANGE union CATEGORICAL columns.
    _CONTENT_COLS = None

    @staticmethod
    def ContentColumns():
        if JoinOrderBenchmark._CONTENT_COLS is None:
            JoinOrderBenchmark._CONTENT_COLS = {
                '{}.csv'.format(table_name):
                range_cols + JoinOrderBenchmark.CATEGORICAL_COLUMNS[table_name]
                for table_name, range_cols in
                JoinOrderBenchmark.RANGE_COLUMNS.items()
            }
            # Add join keys.
            for table_name in JoinOrderBenchmark._CONTENT_COLS:
                cols = JoinOrderBenchmark._CONTENT_COLS[table_name]
                if table_name == 'title.csv':
                    cols.append('id')
                elif 'movie_id' in JoinOrderBenchmark.BASE_TABLE_PRED_COLS[
                        table_name]:
                    cols.append('movie_id')

        return JoinOrderBenchmark._CONTENT_COLS

    @ staticmethod
    def GeneralColumns():
        content_columns = JoinOrderBenchmark.ContentColumns()
        return {table_name: list(utils.util.unique(content_columns.get(table_name, []) + base_columns)) for table_name, base_columns in JoinOrderBenchmark.BASE_TABLE_PRED_COLS.items()}


    @staticmethod
    def GetFullOuterCardinalityOrFail(join_tables):
        key = tuple(sorted(join_tables))
        return JoinOrderBenchmark.TRUE_FULL_OUTER_CARDINALITY[key]

    @staticmethod
    def GetJobLightJoinKeys():
        # TODO: dynamic implementation to adapt JOB-light-ranges!
        return {
            'movie_companies': 'movie_id',
            'movie_keyword': 'movie_id',
            'title': 'id',
            'cast_info': 'movie_id',
            'movie_info': 'movie_id',
            'movie_info_idx': 'movie_id',
        }


def LoadImdb(table=None,
             data_dir='./datasets/job/',
             try_load_parsed=True,
             use_cols='simple',
             tag=''):
    """Loads IMDB tables with a specified set of columns.
    use_cols:
      simple: only movie_id join keys (JOB-light)
      content: + content columns (JOB-light-ranges)
      multi: all join keys in JOB-M
      full: all join keys in JOB-full
      None: load all columns
    Returns:
      A single CsvTable if 'table' is specified, else a dict of CsvTables.
    """
    assert use_cols in ['simple', 'content', 'multi', 'full', 'general', None], use_cols

    def TryLoad(table_name, filepath, use_cols, **kwargs):
        """Try load from previously parsed (table, columns)."""
        if use_cols:
            cols_str = '-'.join(use_cols)
            parsed_path = filepath[:-4] + '.{}.table'.format(cols_str)
        else:
            parsed_path = filepath[:-4] + '.table'
        if try_load_parsed:
            if os.path.exists(parsed_path):
                arr = np.load(parsed_path, allow_pickle=True)
                print('Loaded parsed Table from', parsed_path)
                table = arr.item()
                print(table)
                return table
        table = common.CsvTable(
            table_name,
            filepath,
            cols=use_cols,
            **kwargs,
        )
        if try_load_parsed:
            np.save(open(parsed_path, 'wb'), table)
            print('Saved parsed Table to', parsed_path)
        return table

    def get_use_cols(filepath):
        if use_cols == 'simple':
            return JoinOrderBenchmark.BASE_TABLE_PRED_COLS.get(filepath, None)
        elif use_cols == 'content':
            return JoinOrderBenchmark.ContentColumns().get(filepath, None)
        # elif use_cols == 'content':
        #     return JoinOrderBenchmark.BASE_TABLE_PRED_COLS.get(filepath, None)  # 修改
        elif use_cols == 'multi':
            return JoinOrderBenchmark.JOB_M_PRED_COLS.get(filepath, None)
        elif use_cols == 'full':
            return JoinOrderBenchmark.JOB_FULL_PRED_COLS.get(filepath, None)
        elif use_cols == 'general':
            return JoinOrderBenchmark.GeneralColumns().get(filepath, None)
        return None  # Load all.

    if isinstance(table, str):
        table = TryLoad(
            table,
            data_dir + table + tag + '.csv',
            use_cols=get_use_cols(table + '.csv'),
            type_casts={},
            escapechar='\\'
        )
        return table
    elif isinstance(table, list):
        tables = {}
        for table_name in table:
            tables[table_name] = TryLoad(
                table_name,
                data_dir + table_name + tag + '.csv',
                use_cols=get_use_cols(table_name + '.csv'),
                type_casts={},
                escapechar='\\',
                )

        return tables

    tables = {}
    for filepath in JoinOrderBenchmark.BASE_TABLE_PRED_COLS:
        tables[filepath[0:-4]] = TryLoad(
            filepath[0:-4],
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
            escapechar='\\',
            )

    return tables


def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv', file_path='./datasets/{}',
            exclude=['Reg Valid Date']):
    csv_file = file_path.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    table = pd.read_csv(csv_file, low_memory=False)
    cols = table.columns
    type_casts = {'Reg Valid Date': np.datetime64}
    for ex in exclude:
        if ex in type_casts.keys():
            type_casts.pop(ex)
        if ex in list(cols):
            cols = cols.delete(list(cols).index('Reg Valid Date'))
    key = list(cols).index(key)
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    tab = common.CsvTable('DMV', csv_file, cols, table, type_casts)
    # tab.key = key
    return tab
