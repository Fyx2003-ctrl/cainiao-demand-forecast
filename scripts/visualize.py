# -*- coding: utf-8 -*-
"""
菜鸟供应链 - 可视化美化版 (专业商务图表)
========================================
用专业配色+数据标签+清晰标题, 让图表有展示价值(作品集/GitHub用)

用法: python visualize.py (需先跑过 eda.py/predict.py)
"""
import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ===== 专业商务配色 =====
COLORS = ['#4E148B', '#FF6900', '#1a3a6b', '#2ecc71', '#e74c3c',
          '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22']

# ===== 全局样式(商务风) =====
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
rcParams['axes.unicode_minus'] = False
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = '#fafbfc'
rcParams['axes.grid'] = True
rcParams['grid.color'] = '#e0e0e0'
rcParams['grid.alpha'] = 0.6
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'cainiao-master', 'data', 'train_all.csv')
OUT_DIR = os.path.join(BASE_DIR, '输出')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print('=' * 50)
    print('菜鸟供应链 - 商务可视化')
    print('=' * 50)

    df = pd.read_csv(DATA_PATH, nrows=100000)
    print(f'数据: {len(df):,}行 × {len(df.columns)}列')

    # ===== 图1: 需求量分布(带长尾标注) =====
    print('\n[1] 需求量分布(美化)...')
    fig, ax = plt.subplots(figsize=(10, 5))
    # 截断极端值看主体分布(0-200)
    df_sub = df[df['label'] <= 200]
    ax.hist(df_sub['label'], bins=40, color=COLORS[0], edgecolor='white', alpha=0.9)
    ax.axvline(df['label'].mean(), color=COLORS[1], linestyle='--', linewidth=2,
               label=f'平均需求量 {df["label"].mean():.1f}')
    ax.axvline(df['label'].median(), color=COLORS[2], linestyle='--', linewidth=2,
               label=f'中位数 {df["label"].median():.0f}')
    ax.set_title('商品需求量分布（长尾特征）', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('需求量', fontsize=11)
    ax.set_ylabel('商品数', fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    # 标注长尾
    ax.annotate('大部分商品需求低\n(长尾)', xy=(150, ax.get_ylim()[1]*0.6),
                fontsize=10, color=COLORS[1], ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '1_需求分布.png'), dpi=200)
    print('  ✅ 输出/1_需求分布.png (长尾标注+均值/中位数)')

    # ===== 图2: 特征相关性Top10(横向条形, 美化) =====
    print('\n[2] 特征相关性Top10(美化)...')
    key_cols = ['label', 'num_alipay_1', 'num_gmv_1', 'num_gmv_sum_1', 'qty_gmv_1',
                'qty_gmv_sum_1', 'pv_ipv_1', 'amt_alipay_1', 'collect_uv_1', 'cart_ipv_1']
    key_cols = [c for c in key_cols if c in df.columns]
    corr = df[key_cols].corr()['label'].drop('label').sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS[0] if v > 0 else COLORS[4] for v in corr.values]
    bars = ax.barh(corr.index, corr.values, color=colors, alpha=0.9)
    ax.set_title('与需求量的相关性 Top9', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('相关系数', fontsize=11)
    # 数据标签
    for bar, val in zip(bars, corr.values):
        ax.text(val + 0.01 if val > 0 else val - 0.03, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    # 重命名可读
    labels = {'num_alipay_1': '1天前支付笔数', 'num_gmv_1': '1天前成交笔数',
              'num_gmv_sum_1': '累计成交笔数', 'qty_gmv_1': '1天前成交量',
              'qty_gmv_sum_1': '累计成交量', 'pv_ipv_1': '1天前浏览量',
              'amt_alipay_1': '1天前支付金额', 'collect_uv_1': '收藏人数',
              'cart_ipv_1': '加购量'}
    ax.set_yticklabels([labels.get(x.get_text(), x.get_text()) for x in ax.get_yticklabels()])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '2_特征相关性.png'), dpi=200)
    print('  ✅ 输出/2_特征相关性.png (中文标签+数据标注)')

    # ===== 图3: 模型效果对比(柱状+标注) =====
    print('\n[3] 模型效果对比(美化)...')
    models = ['线性回归\n(全特征)', '线性回归\n(特征选择)', '随机森林\n(特征选择)']
    r2s = [-0.449, 0.708, 0.702]
    colors3 = [COLORS[4], COLORS[0], COLORS[1]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, r2s, color=colors3, width=0.5, alpha=0.9)
    ax.axhline(0, color='#333', linewidth=1)
    ax.set_title('特征工程前后模型效果对比 (R²)', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_ylim(-0.6, 1.0)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03 if val >= 0 else val - 0.08,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    # 箭头标注提升
    ax.annotate('特征选择\n提升1.16', xy=(1, 0.708), xytext=(0.5, 0.85),
                fontsize=10, color=COLORS[0],
                arrowprops=dict(arrowstyle='->', color=COLORS[0]))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '4_模型对比.png'), dpi=200)
    print('  ✅ 输出/4_模型对比.png (对比+提升标注)')

    # ===== 图4: 特征重要性Top10(美化) =====
    print('\n[4] 特征重要性Top10(美化)...')
    # 用之前predict的特征重要性(简化: 用相关性替代展示)
    feature_cols = [c for c in df.columns if c not in ['label', 'item_id', 'store_code']]
    X = df[feature_cols].fillna(0)
    y = df['label']
    corr_all = X.corrwith(y).abs().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(corr_all.index[::-1], corr_all.values[::-1], color=COLORS[0], alpha=0.9)
    ax.set_title('需求量预测 关键特征 Top10', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('与需求量相关性(绝对值)', fontsize=11)
    for bar, val in zip(bars, corr_all.values[::-1]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '3_特征重要性.png'), dpi=200)
    print('  ✅ 输出/3_特征重要性.png')

    print('\n' + '=' * 50)
    print('✅ 可视化完成! 4张商务图表在 输出/ 目录')
    print('=' * 50)


if __name__ == '__main__':
    main()
