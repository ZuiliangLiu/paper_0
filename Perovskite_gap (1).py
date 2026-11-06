
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


GAP_THRESHOLD = 0.2          # eV；按论文采用严格大于
CLASS_PROB_THRESHOLD = 0.5  # 分类概率阈值，与带隙阈值不同

NUMERIC_COLS = [
    'N1', 'S1', 'D1', 'IE1', 'X1', 'R1',
    'N2', 'S2', 'D2', 'IE2', 'X2', 'R2',
    'N3', 'S3', 'D3', 'IE3', 'X3', 'R3',
    'Volume_per_atom', 'Density (g/cm³)'
]
CATEGORICAL_COLS = ['Space Group Number', 'Is Metal']
INPUT_COLS = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL = 'Band Gap (eV)'

XGB_PARAMS = dict(
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


def make_preprocessor():
    # 两个模型分别创建预处理器，各自只在对应训练数据上拟合。
    return ColumnTransformer(
        transformers=[
            ('numeric', MinMaxScaler(), NUMERIC_COLS),
            (
                'categorical',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                CATEGORICAL_COLS
            )
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )


def make_inputs(df):
    data = df.copy()
    data['Volume_per_atom'] = data['Volume (Å³)'] / data['Sites']
    missing = [col for col in INPUT_COLS if col not in data.columns]
    if missing:
        raise ValueError(f'缺少输入特征：{missing}')
    inputs = data[INPUT_COLS].copy()
    if inputs.isna().any().any():
        raise ValueError('输入特征存在缺失值，请检查 CSV。')
    if not np.isfinite(inputs[NUMERIC_COLS].to_numpy(dtype=float)).all():
        raise ValueError('数值特征包含无穷值，请检查体积和 Sites 等列。')
    return inputs


def main(show_plots=True):
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)

    train_df = pd.read_csv(base_dir / 'train.csv')
    test_df = pd.read_csv(base_dir / 'text.csv')
    predict_df = pd.read_csv(base_dir / 'predict.csv')

    train_inputs = make_inputs(train_df)
    test_inputs = make_inputs(test_df)
    predict_inputs = make_inputs(predict_df)
    train_targets = train_df[TARGET_COL].astype(float)
    test_targets = test_df[TARGET_COL].astype(float)

    # 第一阶段：分类“带隙是否大于 0.2 eV”。
    train_classes = train_targets.gt(GAP_THRESHOLD).astype(int)
    test_classes = test_targets.gt(GAP_THRESHOLD).astype(int)
    if train_classes.nunique() != 2 or test_classes.nunique() != 2:
        raise ValueError('分类训练及 ROC 测试都需要同时包含阈值两侧的样本。')

    classifier = Pipeline([
        ('preprocessor', make_preprocessor()),
        ('xgboost', XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            **XGB_PARAMS
        ))
    ])
    classifier.fit(train_inputs, train_classes)

    test_probability = classifier.predict_proba(test_inputs)[:, 1]
    test_predicted_classes = (test_probability >= CLASS_PROB_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        test_classes, test_predicted_classes, labels=[0, 1]
    ).ravel()
    auc = roc_auc_score(test_classes, test_probability)
    accuracy = accuracy_score(test_classes, test_predicted_classes)
    precision = precision_score(test_classes, test_predicted_classes, zero_division=0)
    recall = recall_score(test_classes, test_predicted_classes, zero_division=0)
    specificity = float(tn / (tn + fp))

    print('第一阶段：Eg > 0.2 eV 分类模型，完整测试集')
    print(f'AUC：{auc:.4f}')
    print(f'Accuracy：{accuracy:.4f}')
    print(f'Precision：{precision:.4f}')
    print(f'Recall：{recall:.4f}')
    print(f'Specificity：{specificity:.4f}')

    # 第二阶段：用真实标签筛选回归训练/测试子集。
    # 不用分类器对训练集的拟合预测来代替真实带隙筛选。
    regression_train_mask = train_targets.gt(GAP_THRESHOLD)
    regression_test_mask = test_targets.gt(GAP_THRESHOLD)
    if regression_train_mask.sum() < 2 or regression_test_mask.sum() < 2:
        raise ValueError('阈值以上的回归训练、测试样本不足。')

    regressor = Pipeline([
        ('preprocessor', make_preprocessor()),
        ('xgboost', XGBRegressor(
            objective='reg:squarederror', **XGB_PARAMS
        ))
    ])
    regressor.fit(
        train_inputs.loc[regression_train_mask],
        train_targets.loc[regression_train_mask]
    )

    y_true = test_targets.loc[regression_test_mask].to_numpy()
    y_pred = regressor.predict(test_inputs.loc[regression_test_mask])
    regression_r2 = float(r2_score(y_true, y_pred))
    regression_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    print('\n第二阶段：回归模型，真实 Eg > 0.2 eV 的测试子集')
    print(f'回归训练样本：{int(regression_train_mask.sum())}')
    print(f'回归测试样本：{int(regression_test_mask.sum())}')
    print(f'R²：{regression_r2:.4f}')
    print(f'RMSE：{regression_rmse:.4f} eV')

    # 无标签预测集：先分类筛选，通过者再回归预测数值。
    predict_probability = classifier.predict_proba(predict_inputs)[:, 1]
    predict_class_pass = predict_probability >= CLASS_PROB_THRESHOLD
    predicted_gaps = np.full(len(predict_df), np.nan)
    if predict_class_pass.any():
        predicted_gaps[predict_class_pass] = regressor.predict(
            predict_inputs.loc[predict_class_pass]
        )
    final_pass = predict_class_pass & (predicted_gaps > GAP_THRESHOLD)

    candidates = predict_df.loc[final_pass].copy()
    candidates.insert(0, 'Source CSV Row', np.flatnonzero(final_pass) + 2)
    candidates['P(Eg > 0.2 eV)'] = predict_probability[final_pass]
    candidates['Predicted Band Gap (eV)'] = predicted_gaps[final_pass]
    candidate_path = output_dir / 'predict_candidates_Eg_gt_0p2.csv'
    candidates.to_csv(candidate_path, index=False, encoding='utf-8-sig')

    print('\n预测集筛选结果')
    print(f'原始预测样本：{len(predict_df)}')
    print(f'通过分类筛选：{int(predict_class_pass.sum())}')
    print(f'回归预测 Eg > 0.2 eV：{len(candidates)}')
    print(f'筛选结果已保存：{candidate_path.name}')

    sns.set_theme(style='ticks', font_scale=1.05)

    # 图1：分类器 ROC 曲线，用概率计算，不用 0/1 分类结果计算曲线。
    fpr, tpr, _ = roc_curve(test_classes, test_probability)
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot(fpr, tpr, color='#E34242', linewidth=1.6, label='ROC curve')
    ax1.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.2, label='Random guess')
    ax1.text(
        0.52, 0.23,
        f'AUC = {auc:.3f}\nAccuracy = {accuracy:.3f}\n'
        f'Precision = {precision:.3f}\nRecall = {recall:.3f}',
        transform=ax1.transAxes, fontsize=11
    )
    ax1.set(xlim=(0, 1), ylim=(0, 1.02),
            xlabel='False positive rate', ylabel='True positive rate',
            title='Stage 1: Eg > 0.2 eV classification (test)')
    ax1.legend(loc='lower right')
    fig1.tight_layout()
    fig1.savefig(output_dir / '01_classifier_roc.png', dpi=300, bbox_inches='tight')

    # 图2：与论文回归图对应，横轴是真实带隙，纵轴是回归预测带隙。
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, color='#008B8B', s=38,
                    alpha=0.85, edgecolor='white', linewidth=0.4, ax=ax2)
    lower = min(0.0, float(y_true.min()), float(y_pred.min()))
    upper = max(float(y_true.max()), float(y_pred.max()))
    margin = max((upper - lower) * 0.05, 0.1)
    limits = [lower - margin, upper + margin]
    ax2.plot(limits, limits, 'k--', linewidth=1.2, label='y = x')
    ax2.set_xlim(limits)
    ax2.set_ylim(limits)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel('Actual DFT band gap (eV)')
    ax2.set_ylabel('Predicted band gap (eV)')
    ax2.set_title('Stage 2: regression test (actual Eg > 0.2 eV)')
    ax2.text(0.59, 0.12,
             f'$R^2$ = {regression_r2:.3f}\nRMSE = {regression_rmse:.3f} eV\nN = {len(y_true)}',
             transform=ax2.transAxes, fontsize=11)
    ax2.legend(loc='upper left')
    fig2.tight_layout()
    fig2.savefig(output_dir / '02_regression_test.png', dpi=300, bbox_inches='tight')

    # 图3：predict.csv 没有真实带隙，只画最终候选的预测值分布。
    fig3, ax3 = plt.subplots(figsize=(7, 4.8))
    if not candidates.empty:
        sns.histplot(candidates['Predicted Band Gap (eV)'],
                     bins=min(15, max(5, int(np.sqrt(len(candidates))))),
                     color='#2878B5', edgecolor='white', ax=ax3)
    else:
        ax3.text(0.5, 0.5, 'No candidates passed both stages.',
                 transform=ax3.transAxes, ha='center', va='center')
    ax3.set_xlabel('Predicted band gap (eV)')
    ax3.set_ylabel('Number of candidates')
    ax3.set_title(f'Prediction set: final candidates (N = {len(candidates)})')
    fig3.tight_layout()
    fig3.savefig(output_dir / '03_predict_gap_distribution.png', dpi=300, bbox_inches='tight')

    # 分类与回归指标分别报告；回归测试指标不是整个筛选流程的端到端指标。
    metrics = {
        'gap_threshold_eV': GAP_THRESHOLD,
        'classification_probability_threshold': CLASS_PROB_THRESHOLD,
        'classification_test': {
            'n': len(test_df), 'auc': float(auc), 'accuracy': float(accuracy),
            'precision': float(precision), 'recall': float(recall),
            'specificity': specificity,
            'confusion_matrix_tn_fp_fn_tp': [int(tn), int(fp), int(fn), int(tp)]
        },
        'regression_test_actual_above_threshold': {
            'train_n': int(regression_train_mask.sum()), 'test_n': len(y_true),
            'r2': regression_r2, 'rmse_eV': regression_rmse
        },
        'prediction': {
            'total_n': len(predict_df), 'classifier_pass_n': int(predict_class_pass.sum()),
            'final_candidate_n': len(candidates)
        }
    }
    (output_dir / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    # 单独输出 Sr2GaSbO6 的带隙预测结果
    target_formula = 'Sr2GaSbO6'

    target_result = candidates.loc[
        candidates['Formula'].astype(str).str.strip().eq(target_formula),
        ['Formula', 'Space Group Number', 'Predicted Band Gap (eV)']
    ]

    print('\nSr2GaSbO6 的带隙预测结果：')

    if target_result.empty:
        print('该材料未进入最终候选，请检查 predict.csv 及两阶段筛选结果。')
    else:
        print(target_result.to_string(index=False))


    if show_plots:
        plt.show()
    else:
        plt.close('all')
    return metrics


if __name__ == '__main__':
    main()
