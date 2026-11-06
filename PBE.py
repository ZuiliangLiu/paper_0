from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# 将 Pe_PBE.csv 与本代码放在同一个文件夹中，可直接在 PyCharm 运行。
data_path = Path(__file__).resolve().parent / "Pe_PBE.csv"
pe_df = pd.read_csv(data_path)

# print(pe_df.head())

pe_df.dropna(subset=["band_gap_eV","decomposition_energy_eV_per_formula_unit"],inplace=True)

use_sample = False #不执行抽样
sample_fraction = 0.1
if use_sample:
    pe_df = pe_df.sample(frac=sample_fraction, random_state=42).copy()#抽样开关

train_val_df,test_df = train_test_split(pe_df,test_size=0.2,random_state=42)#20%为测试集
train_df,val_df = train_test_split(train_val_df,test_size=0.25,random_state=42)#总体约60%训练、20%验证、20%测试

input_cols = [train_df.columns[2]] + list(train_df.columns[4:-2])
gap_target_col = "band_gap_eV"
energy_target_col = "decomposition_energy_eV_per_formula_unit"

train_inputs = train_df[input_cols].copy()
train_gap_targets = train_df[gap_target_col].copy()
train_energy_targets = train_df[energy_target_col].copy()

test_inputs = test_df[input_cols].copy()
test_gap_targets = test_df[gap_target_col].copy()
test_energy_targets = test_df[energy_target_col].copy()

val_inputs = val_df[input_cols].copy()
val_gap_targets = val_df[gap_target_col].copy()
val_energy_targets = val_df[energy_target_col].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()#数值列
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()#分类列

# print(train_inputs[numeric_cols].describe())
# print(train_inputs[categorical_cols].nunique())

encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')#sparse_output=False：转换结果输出为普通的 NumPy 数组。
#handle_unknown='ignore'：转换时，如果遇到学习阶段没见过的类别，不报错，该特征对应的编码列全部填 0
encoder.fit(train_inputs[categorical_cols])#只在训练集上学习类别
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))#保存这些列名
#分类列中的值转换成独热编码，然后把编码结果添加到新的列中
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


gap_model = RandomForestRegressor(
    n_jobs=-1,
    random_state=42
)

energy_model = RandomForestRegressor(
    n_jobs=-1,
    random_state=42
)

# 两个模型分别学习带隙、分解能，共用相同的数据划分和输入特征。
gap_model.fit(X_train, train_gap_targets)
energy_model.fit(X_train, train_energy_targets)

train_gap_preds = gap_model.predict(X_train)
val_gap_preds = gap_model.predict(X_val)
test_gap_preds = gap_model.predict(X_test)

train_energy_preds = energy_model.predict(X_train)
val_energy_preds = energy_model.predict(X_val)
test_energy_preds = energy_model.predict(X_test)

gap_rmse = root_mean_squared_error(val_gap_targets, val_gap_preds)
energy_rmse = root_mean_squared_error(val_energy_targets, val_energy_preds)

print("带隙模型：")
print(f"训练集 RMSE：{root_mean_squared_error(train_gap_targets, train_gap_preds):.4f} eV")
print(f"验证集 RMSE：{gap_rmse:.4f} eV")
print(f"测试集 RMSE：{root_mean_squared_error(test_gap_targets, test_gap_preds):.4f} eV")

print("\n分解能模型：")
print(f"训练集 RMSE：{root_mean_squared_error(train_energy_targets, train_energy_preds):.4f} eV/f.u.")
print(f"验证集 RMSE：{energy_rmse:.4f} eV/f.u.")
print(f"测试集 RMSE：{root_mean_squared_error(test_energy_targets, test_energy_preds):.4f} eV/f.u.")


plt.figure(figsize=(6, 6))

# 横轴：验证集中的DFT带隙；纵轴：模型预测带隙
sns.regplot(
    x=val_gap_targets,
    y=val_gap_preds,
    ci=None,  # 不显示拟合线的置信区间
    scatter_kws={"s": 40, "alpha": 0.7, "color": "steelblue"},
    line_kws={"color": "red", "label": "Linear fit"}
)

# 确定横纵轴共同的显示范围
lower = min(val_gap_targets.min(), val_gap_preds.min())
upper = max(val_gap_targets.max(), val_gap_preds.max())
margin = max((upper - lower) * 0.05, 0.01)

# 理想预测参考线：预测值等于实际值
plt.plot(
    [lower, upper], [lower, upper],
    "k--", label="Ideal: y = x"
)

plt.xlim(lower - margin, upper + margin)
plt.ylim(lower - margin, upper + margin)
plt.gca().set_aspect("equal", adjustable="box")

plt.xlabel("DFT band gap (eV)")
plt.ylabel("Predicted band gap (eV)")
plt.title(f"Random Forest - Band Gap (PBE)\nValidation RMSE = {gap_rmse:.3f} eV")
plt.legend()
plt.tight_layout()
# 此处先不 show，最后统一显示两张图。
plt.figure(figsize=(6, 6))

# 横轴：验证集中的DFT分解能；纵轴：分解能模型的预测值。
sns.regplot(
    x=val_energy_targets,
    y=val_energy_preds,
    ci=None,
    scatter_kws={"s": 40, "alpha": 0.7, "color": "steelblue"},
    line_kws={"color": "red", "label": "Linear fit"}
)

# 同时覆盖实际值、预测值的范围，包括负值
lower = min(val_energy_targets.min(), val_energy_preds.min())
upper = max(val_energy_targets.max(), val_energy_preds.max())
margin = max((upper - lower) * 0.05, 0.01)

# 理想预测线
plt.plot(
    [lower, upper], [lower, upper],
    "k--", label="Ideal: y = x"
)

plt.xlim(lower - margin, upper + margin)
plt.ylim(lower - margin, upper + margin)
plt.gca().set_aspect("equal", adjustable="box")

plt.xlabel("DFT decomposition energy (eV/f.u.)")
plt.ylabel("Predicted decomposition energy (eV/f.u.)")
plt.title(f"Random Forest - Decomposition Energy (PBE)\nValidation RMSE = {energy_rmse:.3f} eV/f.u.")
plt.legend()
plt.tight_layout()
new_inputs = pd.DataFrame([{
    input_cols[0]: "Tetra",  # 本CSV中的四方相名称为Tetra，需与训练数据保持一致。

    # A 位组成
    "K": 0.0,
    "Rb": 0.0,
    "Cs": 1.0,
    "MA": 0.0,
    "FA": 0.0,

    # B 位组成
    "Ca": 0.25,
    "Sr": 0.0,
    "Ba": 0.0,
    "Ge": 0.25,
    "Sn": 0.0,
    "Pb": 0.50,

    # X 位组成
    "Cl": 0.0,
    "Br": 1.0,
    "I": 0.0
}])

# 保持原始输入列与训练时一致
new_inputs = new_inputs[input_cols].copy()

# 使用训练时已经拟合好的编码器处理晶体结构
encoded_values = encoder.transform(new_inputs[categorical_cols])

# 兼容稀疏矩阵输出
if hasattr(encoded_values, "toarray"):
    encoded_values = encoded_values.toarray()

encoded_names = encoder.get_feature_names_out(categorical_cols)
new_inputs[encoded_names] = encoded_values

# 保证最终特征及排列顺序与模型训练时完全一致
X_new = new_inputs[X_train.columns]

# 分别预测同一种材料的PBE带隙与PBE分解能。
gap_prediction = gap_model.predict(X_new)[0]
energy_prediction = energy_model.predict(X_new)[0]

print("\n预测材料：CsCa0.25Ge0.25Pb0.50Br3")
print(f"预测带隙：{gap_prediction:.4f} eV")
print(f"预测分解能：{energy_prediction:.4f} eV/f.u.")

plt.show()
