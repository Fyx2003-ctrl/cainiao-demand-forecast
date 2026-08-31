# -*- coding: utf-8 -*-
"""
菜鸟供应链需求量预测 v2 (特征选择版)
====================================
v2改进:
1. 特征选择: 只用与label相关性高的Top30特征 (去冗余, 提升效果)
2. 修复RMSE弃用警告 (用root_mean_squared_error)
3. 线性回归 + 随机森林对比

用法: python predict.py
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'cainiao-master', 'data', 'train_all.csv')
OUT_DIR = os.path.join(BASE_DIR, '输出')
os.makedirs(OUT_DIR, exist_ok=True)

TOP_N = 30  # 保留特征数


def main():
    print('=' * 50)
    print('菜鸟供应链需求量预测 v2 (特征选择)')
    print('=' * 50)

    print('\n[1] 读取数据...')
    df = pd.read_csv(DATA_PATH, nrows=50000)
    print(f'  样本数: {len(df):,} 行 / 特征: {len(df.columns)} 列')

    print('\n[2] 特征选择 (按与label相关性Top%d)...' % TOP_N)
    exclude = ['label', 'item_id', 'store_code']
    feature_cols = [c for c in df.columns if c not in exclude]
    X_all = df[feature_cols].fillna(0)
    y = df['label']

    # 计算每个特征与label的相关性, 取Top30
    corr = X_all.corrwith(y).abs().sort_values(ascending=False)
    top_features = corr.head(TOP_N).index.tolist()
    print(f'  选用特征: {top_features[:5]}... 等{len(top_features)}个')

    X = X_all[top_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f'  训练集: {len(X_train):,} / 测试集: {len(X_test):,}')

    print('\n[3] 训练线性回归...')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_r2 = r2_score(y_test, y_pred_lr)
    lr_rmse = root_mean_squared_error(y_test, y_pred_lr)
    print(f'  R² = {lr_r2:.3f}, RMSE = {lr_rmse:.2f}')

    print('\n[4] 训练随机森林...')
    rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_rmse = root_mean_squared_error(y_test, y_pred_rf)
    print(f'  R² = {rf_r2:.3f}, RMSE = {rf_rmse:.2f}')

    print('\n[5] 特征重要性TOP10:')
    importance = pd.Series(rf.feature_importances_, index=top_features)
    top10 = importance.sort_values(ascending=False).head(10)
    for k, v in top10.items():
        print(f'  {k}: {v:.4f}')

    plt.figure(figsize=(9, 6))
    top10.sort_values().plot(kind='barh', color='#FF6900')
    plt.title('随机森林特征重要性 TOP10 (特征选择后)')
    plt.xlabel('重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '3_特征重要性.png'), dpi=150)
    print('  ✅ 输出/3_特征重要性.png')

    print('\n[6] 模型对比图...')
    models = ['线性回归', '随机森林']
    r2s = [lr_r2, rf_r2]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, r2s, color=['#4E148B', '#FF6900'], width=0.5)
    plt.title('模型效果对比 (R², 特征选择后)')
    plt.ylabel('R²')
    plt.ylim(0, 1)
    for bar, val in zip(bars, r2s):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '4_模型对比.png'), dpi=150)
    print('  ✅ 输出/4_模型对比.png')

    print('\n' + '=' * 50)
    print('✅ 预测完成!')
    print(f'  线性回归 R² = {lr_r2:.3f} (特征选择后应转正)')
    print(f'  随机森林 R² = {rf_r2:.3f} (应提升到0.8+)')
    print(f'  最佳模型: {"随机森林" if rf_r2 > lr_r2 else "线性回归"}')
    print('=' * 50)


if __name__ == '__main__':
    main()
