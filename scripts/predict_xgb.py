# -*- coding: utf-8 -*-
"""
菜鸟供应链需求量预测 v3 (XGBoost 进阶)
=======================================
XGBoost = 梯度提升树 (比随机森林更强, 业界主流)

用法: python predict_xgb.py
需要: pip install xgboost
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
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'cainiao-master', 'data', 'train_all.csv')
OUT_DIR = os.path.join(BASE_DIR, '输出')
os.makedirs(OUT_DIR, exist_ok=True)

TOP_N = 30


def main():
    print('=' * 50)
    print('菜鸟供应链需求量预测 v3 (XGBoost)')
    print('=' * 50)

    df = pd.read_csv(DATA_PATH, nrows=50000)
    print(f'数据: {len(df):,}行 × {len(df.columns)}列')

    exclude = ['label', 'item_id', 'store_code']
    feature_cols = [c for c in df.columns if c not in exclude]
    X_all = df[feature_cols].fillna(0)
    y = df['label']

    # 特征选择 Top30
    corr = X_all.corrwith(y).abs().sort_values(ascending=False)
    top_features = corr.head(TOP_N).index.tolist()
    X = X_all[top_features]
    print(f'特征选择: Top{TOP_N}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f'训练: {len(X_train):,} / 测试: {len(X_test):,}')

    # 1. 线性回归 (基线)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_r2 = r2_score(y_test, lr.predict(X_test))

    # 2. 随机森林
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test))

    # 3. XGBoost
    print('\n训练 XGBoost...')
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                           random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        xgb_r2 = r2_score(y_test, y_pred_xgb)
        xgb_rmse = root_mean_squared_error(y_test, y_pred_xgb)
        print(f'  XGBoost R² = {xgb_r2:.3f}, RMSE = {xgb_rmse:.2f}')
    except ImportError:
        print('  ⚠️ 无xgboost, 运行: pip install xgboost')
        xgb_r2 = None

    print(f'\n线性回归 R² = {lr_r2:.3f}')
    print(f'随机森林 R² = {rf_r2:.3f}')
    if xgb_r2 is not None:
        print(f'XGBoost  R² = {xgb_r2:.3f}')

    # 模型对比图
    models = ['线性回归', '随机森林'] + (['XGBoost'] if xgb_r2 is not None else [])
    r2s = [lr_r2, rf_r2] + ([xgb_r2] if xgb_r2 is not None else [])
    colors = ['#4E148B', '#FF6900', '#1a3a6b']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, r2s, color=colors[:len(models)], width=0.5)
    plt.title('模型效果对比 (R², 特征选择后)', fontsize=13, fontweight='bold')
    plt.ylabel('R²')
    plt.ylim(0, 1)
    for bar, val in zip(bars, r2s):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f'{val:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '4_模型对比.png'), dpi=200)
    print('\n✅ 已更新 输出/4_模型对比.png (含XGBoost)')


if __name__ == '__main__':
    main()
