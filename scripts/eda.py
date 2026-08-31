# -*- coding: utf-8 -*-
"""
菜鸟供应链数据探索分析 (EDA)
================================
功能:
1. 读取数据, 查看结构与规模
2. 目标变量(需求量label)分布
3. 关键特征与需求量的相关性
4. 输出可视化图表到 输出/ 目录

用法: python eda.py
数据: 需与脚本同目录的 cainiao_data/ 下 (train_all.csv)
"""
import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径: 脚本同级 cainiao_data/train_all.csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'cainiao-master', 'data', 'train_all.csv')
OUT_DIR = os.path.join(BASE_DIR, '输出')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print('=' * 50)
    print('菜鸟供应链数据探索分析 (EDA)')
    print('=' * 50)

    print('\n[1] 读取数据...')
    df = pd.read_csv(DATA_PATH, nrows=100000)
    print(f'  样本数: {len(df):,} 行')
    print(f'  特征数: {len(df.columns)} 列')

    print('\n[2] 数据概览:')
    print(f'  商品数: {df["item_id"].nunique():,}')
    print(f'  店铺数: {df["store_code"].nunique():,}')
    print(f'  需求量范围: {df["label"].min():.0f} ~ {df["label"].max():.0f}')
    print(f'  需求量均值: {df["label"].mean():.1f}')

    print('\n[3] 生成需求分布图...')
    plt.figure(figsize=(8, 4))
    df['label'].hist(bins=50, color='#4E148B', edgecolor='white')
    plt.title('商品需求量分布')
    plt.xlabel('需求量')
    plt.ylabel('商品数')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '1_需求分布.png'), dpi=150)
    print('  ✅ 输出/1_需求分布.png')

    print('\n[4] 计算特征相关性...')
    key_cols = ['label', 'num_gmv_1', 'qty_gmv_1', 'amt_alipay_1', 'num_alipay_1',
                'pv_ipv_1', 'cart_ipv_1', 'collect_uv_1',
                'num_gmv_sum_1', 'qty_gmv_sum_1']
    key_cols = [c for c in key_cols if c in df.columns]
    corr = df[key_cols].corr()['label'].sort_values(ascending=False)
    print('  与需求量的相关性:')
    for k, v in corr.items():
        print(f'    {k}: {v:.3f}')

    print('\n[5] 生成相关性热力图...')
    try:
        import seaborn as sns
        plt.figure(figsize=(9, 7))
        sns.heatmap(df[key_cols].corr(), annot=True, cmap='RdBu_r', fmt='.2f',
                    xticklabels=key_cols, yticklabels=key_cols)
        plt.title('关键特征相关性热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, '2_相关性热力图.png'), dpi=150)
        print('  ✅ 输出/2_相关性热力图.png')
    except ImportError:
        print('  ⚠️ 无seaborn, 跳过 (pip install seaborn)')

    print('\n' + '=' * 50)
    print('✅ EDA完成! 图表在 输出/ 目录')
    print('=' * 50)


if __name__ == '__main__':
    main()
