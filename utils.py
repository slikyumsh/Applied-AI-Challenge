import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class Paths:
    train_ids_target = '../data/train/ids_target.parquet.fastparquet'
    train_egrul = '../data/train/prelead_features_egrul.parquet.fastparquet'
    train_revenue = '../data/train/prelead_features_revenue.parquet.fastparquet'
    train_user_info = '../data/train/prelead_features_user_info.parquet.fastparquet'
    train_goszakupki = '../data/train/prelead_features_goszakupki.parquet.fastparquet'
    train_client_gender = '../data/train/prelead_features_client_gender.parquet.fastparquet'
    train_contractors_banks = '../data/train/prelead_features_contractors_banks.parquet.fastparquet'
    train_client_history = '../data/train/prelead_features_client_history.parquet.fastparquet'
    train_constractors_directional = '../data/train/prelead_features_contractors_directional.parquet.fastparquet'
    train_holding = '../data/train/prelead_features_holding.parquet.fastparquet'
    train_time = '../data/train/prelead_features_time_features.parquet.fastparquet'

    test_ids_target = '../data/test/ids_target.parquet.fastparquet'
    test_egrul = '../data/test/prelead_features_egrul.parquet.fastparquet'
    test_revenue = '../data/test/prelead_features_revenue.parquet.fastparquet'
    test_user_info = '../data/test/prelead_features_user_info.parquet.fastparquet'
    test_goszakupki = '../data/test/prelead_features_goszakupki.parquet.fastparquet'
    test_client_gender = '../data/test/prelead_features_client_gender.parquet.fastparquet'
    test_contractors_banks = '../data/test/prelead_features_contractors_banks.parquet.fastparquet'
    test_client_history = '../data/test/prelead_features_client_history.parquet.fastparquet'
    test_constractors_directional = '../data/test/prelead_features_contractors_directional.parquet.fastparquet'
    test_holding = '../data/test/prelead_features_holding.parquet.fastparquet'
    test_time = '../data/test/prelead_features_time_features.parquet.fastparquet'



def load_data(paths):
    dataframes = []
    attributes = [getattr(paths, attr) for attr in dir(paths) if not attr.startswith("__")]
    
    # Читаем каждый файл и добавляем его содержимое в список датафреймов
    for path in attributes:
        df = pd.read_parquet(path)
        dataframes.append(df)
    
    return dataframes


def load_and_join_data(paths):
    # Инициализация списков для хранения датафреймов
    train_dfs = []
    test_dfs = []
    
    # Обходим атрибуты, загружаем датафреймы и добавляем их в соответствующие списки
    for attr in dir(paths):
        if attr.startswith("__") or not attr.startswith(('train', 'test')):
            continue  # Пропускаем специальные атрибуты и не относящиеся к данным
        
        path = getattr(paths, attr)
        df = pd.read_parquet(path)
        
        if "train" in attr:
            train_dfs.append(df)
        elif "test" in attr:
            test_dfs.append(df)

    base_train_df = train_dfs.pop(0)
    base_test_df = test_dfs.pop(0)
    
    for df in train_dfs:
        base_train_df = base_train_df.merge(df, on='id', how='left')
    
    for df in test_dfs:
        base_test_df = base_test_df.merge(df, on='id', how='left')
    
    return base_train_df, base_test_df


def code_gender(x):
    if x == 'женщина':
        return 0
    if x == 'мужчина':
        return 1
    return -1

def code_role(x):
    if x == 'unknown':
        return 6
    return (int)(x.split('_')[-1])

def code_grade(x):
    if x == 'unknown':
        return 3
    return (int)(x.split('_')[-1])


def code_tochka_contractors_count_sum_rub_incoming(x):
    if x == '0':
        return 0.001
    if x == '1 - 5':
        return 0.1
    if x == '5 - 10':
        return 0.5
    if x == '10 - 100':
        return 1.
    return 2.

def code_tochka_contractors_avg_sum_rub_incoming(x):
    if x == '< 10000':
        return 0.001
    if x == '10000 - 100000':
        return 0.01
    if x == '100000 - 1000000':
        return 0.1
    return 1.

def code_tochka_constractors_n_ongoing(x):
    if x == '0':
        return 0.001

    if x == '1 - 5':
        return 0.1

    if x == '5 - 10':
        return 0.5

    if x == '10 - 100':
        return 1.

    return 2.


def code_tochka_contractors_avg_sum_rub_outgoing(x):
    if x == '< 10000':
        return 0.001

    if x == '10000 - 100000':
        return 0.01

    if x == '100000 - 1000000':
        return 0.1

    return 1.

def code_revenue(x):
    if x == 'unknown':
        return 0.001
    if x == '<10000000':
        return 0.01
    if x == '10000000 - 50000000':
        return 0.05
    if x == '50000000 - 100000000':
        return 0.1
    return  1.


def code_egrul_reg_months_ago(x):
    if x == '0':
        return 0.01
    if x == '0 - 6':
        return 0.05
    if x == '6 - 12':
        return 0.1
    if x == '12 - 24':
        return 0.2
    if x == '24 - 36':
        return 0.3
    return 0.5

def code_position(x):
    if x == 'unknown':
        return -3.     #кинем среднее, кажется что роль - чем меньше тем выше
    position = (float)(x.split('_')[-1])
    return position

def code_employed_days(x):
    if x < 0:
        return 0.
    return x
    

def code_goszakupki_winner_sum(x):
    if x == '0':
        return 0.00001
    if x == '<100000':
        return 0.0001
    if x == '100000 - 1000000':
        return 0.001
    if x == '1000000 - 10000000':
        return 0.01
    if x == '10000000-1000000000':
        return 0.1
    return 1.


def code_goszakupki_winner_count(x):
    if x == '0':
        return 0.001
    if x == '1 - 5':
        return 0.005
    if x == '5 - 10':
        return 0.01
    if x == '10 - 100':
        return 0.1
    return 1.

def code_timestamp_year(x):
    timestamp = x
    year = timestamp.year
    return year

def code_timestamp_hour(x):
    timestamp = x
    hour = timestamp.hour
    return hour

def code_timestamp_day_of_week(x):
    timestamp = x
    day_of_week = timestamp.weekday()
    return day_of_week

def code_timestamp_season(x):
    timestamp = x
    month = timestamp.month
    if month in [12, 1, 2]:
        season = 0
    elif month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8]:
        season = 2
    else:
        season = 3
    return season


def code_timestamp_is_weekend(x):
    timestamp = x
    day_of_week = timestamp.weekday()  
    weekend = 1 if day_of_week >= 5 else 0
    return weekend

def code_timestamp_part_of_day(x):
    timestamp = x
    hour = timestamp.hour
    if 6 <= hour < 12:
        part_of_day = 0
    elif 12 <= hour < 18:
        part_of_day = 1
    else:
        part_of_day = 2
    return part_of_day

 
def code_goszakupki_total_sum(x):
    if x == '0':
        return 0.00001
    if x == '<100000':
        return 0.0001
    if x == '100000 - 1000000':
        return 0.001
    if x == '1000000 - 10000000':
        return 0.01
    if x == '10000000-1000000000':
        return 0.1
    return 1.

def code_holding_revenue_avg(x):
    if x == 'unknown':
        return 0.00001
    if x == '< 100000':
        return 0.001
    if x == '100000 - 10000000':
        return 0.01
    if x == '10000000 - 100000000':
        return 0.1
    return 1.


def code_goszakupki_total_count(x):
    if x == '0':
        return 0.001
    if x == '1 - 5':
        return 0.005
    if x == '5 - 10':
        return 0.01
    if x == '10 - 100':
        return 0.1
    return 1.


def code_tochka_contractors_count_sum_rub_outgoing(x):
    if x == '0':
        return 0.001
    if x == '1 - 5':
        return 0.005
    if x == '5 - 10':
        return 0.01
    if x == '10 - 100':
        return 0.1
    return 1.

def code_division(x):
    if x == 'unknown':
        return 0
    return (int)(x.split('_')[-1])

def code_time_tz_diff(x):
    return np.abs(x)


def preprocess_dataframe(df):
    df['timestamp_year'] = df['timestamp'].apply(code_timestamp_year)
    #df['timestamp_hour'] = df['timestamp'].apply(code_timestamp_hour)
    df['timestamp_day_of_week'] = df['timestamp'].apply(code_timestamp_day_of_week)
    #df['timestamp_is_weekend'] = df['timestamp'].apply(code_timestamp_is_weekend)
    #df['timestamp_season'] = df['timestamp'].apply(code_timestamp_season)
    #df['timestamp_part_of_day'] = df['timestamp'].apply(code_timestamp_part_of_day)

    # 2. История взаимодействия с клиентом (для простоты предполагаем, что данные уже агрегированы по id)
    df['avg_days_between_takes'] = df['days_since_last_take'] / df['cnt_takes']
    df['success_ratio'] = (df['cnt_takes'] - df['cnt_not_sell']) / df['cnt_takes']
    # 3. Совместимость регионов
    df['division'] = df['division'].apply(code_division)
    df['egrul_region'] = df['egrul_region'].apply(code_division)
    df['region_compatibility'] = (df['division'] == df['egrul_region']).astype(int)

    df['holding_revenue_avg'] = df['holding_revenue_avg'].apply(code_holding_revenue_avg)

    # 5. Совместимость полов
    df['gender_interaction'] = df['seller_gender'] + "_" + df['client_gender']


    df['client_gender'] = df['client_gender'].apply(code_gender)
    df['seller_gender'] = df['seller_gender'].apply(code_gender)


    df['grade'] = df['grade'].apply(code_grade)
    df['role'] = df['role'].apply(code_role)
    df['position'] = df['position'].apply(code_position)
    

    df['employed_days'] = df['employed_days'].apply(code_employed_days)
    df['experience_level'] = pd.cut(df['employed_days'], bins=[-1, 720, 1700, float('inf')], labels=['новичок', 'опытный', 'ветеран'])

   

    df['goszakupki_winner_sum'] = df['goszakupki_winner_sum'].apply(code_goszakupki_winner_sum)
    df['goszakupki_winner_count'] = df['goszakupki_winner_count'].apply(code_goszakupki_winner_count)
    df['goszakupki_total_sum'] = df['goszakupki_total_sum'].apply(code_goszakupki_total_sum)
    df['goszakupki_total_count'] = df['goszakupki_total_count'].apply(code_goszakupki_total_count)

    df['tochka_contractors_count_sum_rub_incoming'] = df['tochka_contractors_count_sum_rub_incoming'].apply(code_tochka_contractors_count_sum_rub_incoming)
    df['tochka_contractors_avg_sum_rub_incoming'] = df['tochka_contractors_avg_sum_rub_incoming'].apply(code_tochka_contractors_count_sum_rub_incoming)
    df['tochka_contractors_n_outgoing'] = df['tochka_contractors_n_outgoing'].apply(code_tochka_constractors_n_ongoing)
    df['tochka_contractors_avg_sum_rub_outgoing'] = df['tochka_contractors_avg_sum_rub_outgoing'].apply(code_tochka_contractors_avg_sum_rub_outgoing)
    
    df['sum_incoming'] = df['tochka_contractors_count_sum_rub_incoming'] * df['tochka_contractors_avg_sum_rub_incoming']
    df['sum_outcoming'] = df['tochka_contractors_n_outgoing'] * df['tochka_contractors_avg_sum_rub_outgoing']

    df['mean_sum_incoming_outcoming'] = df['sum_incoming'] / df['sum_outcoming']
    df['coefficinet_number_of_incoming_outcoming'] = df['tochka_contractors_count_sum_rub_incoming'] / df['tochka_contractors_n_outgoing']
    df['coefficient_mean_sums'] = df['tochka_contractors_avg_sum_rub_incoming'] / df['tochka_contractors_avg_sum_rub_outgoing']


    df['tochka_contractors_count_sum_rub_outgoing'] = df['tochka_contractors_count_sum_rub_outgoing'].apply(code_tochka_contractors_avg_sum_rub_outgoing)

    df['revenue'] = df['revenue'].apply(code_revenue)
    df['time_tz_diff'] = df['time_tz_diff'].fillna(0)
    df['abs_time_diff'] = df['time_tz_diff'].apply(code_time_tz_diff)

    df['egrul_reg_months_ago'] = df['egrul_reg_months_ago'].apply(code_egrul_reg_months_ago)
    df['grade_hack'] = np.where((df['grade'] <= 9) & (df['grade'] >= 5) , 0, 1)
    


    df['total_financial_activity'] = df['tochka_contractors_count_sum_rub_incoming'] + df['tochka_contractors_count_sum_rub_outgoing']
    df['financial_activity_balance'] = df['tochka_contractors_count_sum_rub_incoming'] - df['tochka_contractors_count_sum_rub_outgoing']

    # Участие в госзакупках
    df['win_rate_goszakupki'] = df['goszakupki_winner_count'] / df['goszakupki_total_count']

    # Сравнение размеров компании в холдинге
    df['company_revenue_share'] = df['revenue'] / df['holding_revenue_avg']

    scaler = MinMaxScaler()

    # Выбор столбцов для индекса
    columns_for_index = [
        'tochka_contractors_count_sum_rub_incoming',
        'tochka_contractors_count_sum_rub_outgoing',
        'tochka_contractors_avg_sum_rub_incoming',
        'tochka_contractors_avg_sum_rub_outgoing',
        'goszakupki_winner_sum',
        'goszakupki_winner_count',
        'goszakupki_total_sum',
        'goszakupki_total_count',
        'revenue'
    ]

    # Предполагаем, что данные уже подготовлены и не содержат пропусков
    df[columns_for_index] = scaler.fit_transform(df[columns_for_index])

    # Создание индекса
    df['financial_business_index'] = (
        df['tochka_contractors_count_sum_rub_incoming'] *
        df['tochka_contractors_avg_sum_rub_incoming'] +
        df['tochka_contractors_count_sum_rub_outgoing'] *
        df['tochka_contractors_avg_sum_rub_outgoing']  +
        df['goszakupki_winner_sum'] *  
        df['goszakupki_winner_count']  +
        df['goszakupki_total_sum'] *
        df['goszakupki_total_count'] +
        df['revenue'] 
    ) / 10.

    df = df.drop([
        'timestamp',
        'client_id',
        'seller_id',
        'teamid',
        'hypothesisid',
        'grade_hack',
        'tochka_contractors_count_sum_rub_incoming',
        'tochka_contractors_count_sum_rub_outgoing',
        'tochka_contractors_avg_sum_rub_incoming',
        'tochka_contractors_avg_sum_rub_outgoing',
        'goszakupki_winner_sum',
        'goszakupki_winner_count',
        'goszakupki_total_sum',
        'goszakupki_total_count',
        'time_tz_diff',
        'revenue',
        'cnt_takes',
        'division',

        'holding_cnt_takes', 
        'sum_incoming',
        'total_financial_activity', 
        'financial_activity_balance',
        'is_teamlead'
        ], axis=1)
    

    return df
