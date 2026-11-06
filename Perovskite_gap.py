
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor


def main():
    # CSV 与本脚本放在同一文件夹，PyCharm 工作目录无需另外修改。
    base_dir = Path(__file__).resolve().parent
    train_df = pd.read_csv(base_dir / 'train.csv')
    test_df = pd.read_csv(base_dir / 'text.csv')
    predict_df = pd.read_csv(base_dir / 'predict.csv')

    # 沿用之前的单位原子体积计算，原始 CSV 数值保持不变。
    for df in (train_df, test_df, predict_df):
        df['Volume_per_atom'] = df['Volume (Å³)'] / df['Sites']

    numeric_cols = [
        'N1', 'S1', 'D1', 'IE1', 'X1', 'R1',
        'N2', 'S2', 'D2', 'IE2', 'X2', 'R2',
        'N3', 'S3', 'D3', 'IE3', 'X3', 'R3',
        'Volume_per_atom', 'Density (g/cm³)'
    ]

    # 本次新增：Is Metal 也采用独热编码。
    categorical_cols = ['Space Group Number', 'Is Metal']
    input_cols = numeric_cols + categorical_cols
    target_col = 'Band Gap (eV)'

    train_inputs = train_df[input_cols].copy()
    test_inputs = test_df[input_cols].copy()
    predict_inputs = predict_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    test_targets = test_df[target_col].copy()

    # 数值列归一化；两个分类列独热编码。
    # 所有预处理参数只由训练集学习。
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', MinMaxScaler(), numeric_cols),
            (
                'categorical',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ),
                categorical_cols
            )
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    # 沿用之前的 XGBoost 回归参数。
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('xgboost', xgb_model)
    ])

    # 使用全部训练集训练一次，再在原测试集上评估一次。
    model.fit(train_inputs, train_targets)
    test_predictions = model.predict(test_inputs)
    predict_predictions = model.predict(predict_inputs)

    test_r2 = r2_score(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    print(f'测试集 R²：{test_r2:.4f}')
    print(f'测试集 RMSE：{test_rmse:.4f} eV')
    print(
        f'训练集 R²：'
        f'{r2_score(train_targets, model.predict(train_inputs)):.4f}'
    )

    # predict_predictions 为 predict.csv 对应行顺序的带隙预测结果。
    # 在 PyCharm 调试器中可查看该变量；不改写原始 CSV。

    # 沿用之前的测试集真实值—预测值散点图。
    # 1. 按测试集的预测带隙筛选
    mask = test_predictions >= 0.2

    selected_df = test_df.loc[mask, ['Formula', target_col]].copy()
    selected_df['Predicted Band Gap (eV)'] = test_predictions[mask]

    print(f'\n筛选后保留 {len(selected_df)} 个化合物')

    if selected_df.empty:
        print('没有预测带隙大于等于 0.2 eV 的样本。')
        return

    print(selected_df.to_string(index=False))

    # 2. 获取对应的真实值与预测值
    y_true = selected_df[target_col].to_numpy()
    y_pred = selected_df['Predicted Band Gap (eV)'].to_numpy()

    filtered_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # R²需要至少两个样本，且真实值存在变化
    if len(y_true) >= 2 and np.ptp(y_true) > 0:
        r2_text = f'{r2_score(y_true, y_pred):.3f}'
    else:
        r2_text = 'N/A'

    # 3. 绘制筛选后的散点图
    sns.set_theme(style='ticks')
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.scatterplot(
        x=y_true,
        y=y_pred,
        color='#2878B5',
        s=50,
        alpha=0.75,
        edgecolor='white',
        linewidth=0.5,
        ax=ax
    )

    lower = min(y_true.min(), y_pred.min())
    upper = max(y_true.max(), y_pred.max())
    margin = max((upper - lower) * 0.05, 0.1)
    limits = [lower - margin, upper + margin]

    ax.plot(limits, limits, 'r--', linewidth=1.5, label='y = x')
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Actual band gap (eV)', fontsize=13)
    ax.set_ylabel('Predicted band gap (eV)', fontsize=13)
    ax.set_title('Test set: predicted Eg >= 0.2 eV', fontsize=13)

    ax.text(
        0.05, 0.95,
        f'$R^2$ = {r2_text}\n'
        f'RMSE = {filtered_rmse:.3f} eV\n'
        f'N = {len(y_true)}',
        transform=ax.transAxes,
        va='top',
        fontsize=12
    )

    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
