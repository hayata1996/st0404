import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np

#st.set_page_config(layout="wide")

header = st.container()
dataset = st.container()
eda = st.container()

#st.cache_data
def get_data(filename):
#    taxi_data = pd.read_parquet(filename)
    taxi_data = pd.read_csv(filename)
    return taxi_data

with header:
    st.title('Welcome to my data science project:muscle:')
    st.text('ニューヨークでのタクシー利用状況をplotlyライブラリで可視化します')

with dataset:
    st.header('NYC taxi dataset')
    url = 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page'
    st.write("データの取得元：[NYC web page](%s)" % url)
    st.write("DataFrameの先頭5行を表示、横にスライドできます")
#    taxi_data = get_data('yellow_tripdata_2024-01.parquet')
    oridata = get_data('minidata.csv')
    taxi_data = get_data('newdata.csv')
    st.write(taxi_data.head())
    st.write('中央値などの諸情報')
    st.write(taxi_data.describe())

    target = taxi_data['trip_distance']
    train = taxi_data.drop('trip_distance', axis="columns")
    
    st.text('今回の目的関数であるタクシーでの移動距離の分布')
    fig = px.bar(oridata['trip_distance'])
    st.plotly_chart(fig, use_container_width=True)
    st.text('上側99%で外れ値を削除後')
    fig = px.bar(target)
    st.plotly_chart(fig, use_container_width=True)
    st.text('多すぎて分からないため20万サンプルのうち150データをランダムに表示')
    random = random.sample(range(0, 198000), 150)
    fig = px.bar(target[random])
    st.plotly_chart(fig, use_container_width=True)

# EDA用にクラスを準備
categorical_features = taxi_data.columns[taxi_data.dtypes=='object'].tolist()
numeric_features = taxi_data.columns[taxi_data.dtypes!='object'].tolist()
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0B2447"]

def plot_distribution_pairs(train, test, feature, palette=None):
    data_df = train.copy()
    data_df['set'] = 'train'
    data_df = pd.concat([data_df, test.copy()])
    data_df['set'].fillna('test', inplace=True)
    data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=data_df, x=feature, hue='set', palette=palette, ax=ax)
    ax.set_title(f"Paired train/test distributions of {feature}")
    st.pyplot(fig)

def plot_distribution_pairs_boxplot(train, test, feature, palette=None):
    data_df = train.copy()
    data_df['set'] = 'train'
    data_df = pd.concat([data_df, test.copy()])
    data_df['set'].fillna('test', inplace=True)
    data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=data_df, x='set', y=feature, palette=palette, ax=ax)
    ax.set_title(f"Paired train/test boxplots of {feature}")
    st.pyplot(fig)

with eda:
    st.header("概要が分かったところでEDAをしてみよう")
    for feature in numeric_features:
        plot_distribution_pairs(train, target, feature, palette=color_list)
        plot_distribution_pairs_boxplot(train, target, feature, palette=color_list)

    st.write('まだEDAを始めたばかりですが、')
    st.write('こういったタスクはJupyter notebookかgoogle colabを共有した方がいいですね😅')
    st.write('動きも遅くなるし、他でできること以上のことはできない')