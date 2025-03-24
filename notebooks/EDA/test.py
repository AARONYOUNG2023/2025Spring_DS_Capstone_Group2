import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 定义显著性标注函数
def significance_marker(p_val):
    if p_val < 0.05:
        return "*"
    elif p_val < 0.1:
        return "°"
    else:
        return "n.s."

# 数据整理：每个条件包含 3 个样本
data = {
    'S0':   [0.840775251, 1.041348597, 1.267031855],
    'S2.5': [1.30079653,  1.284156476, 1.447891771],
    'S5':   [1.604879705, 1.32345248,  1.342409364],
    'S10':  [1.695382731, 1.944417514, 2.283764598],
    'S15':  [2.585801964, 2.474860641, 2.336889745]
}

df = pd.DataFrame(data)
conditions = list(data.keys())

# 计算每组的均值与标准误
means = df.mean()
sems = df.sem()

# 绘制柱状图（均值 + 误差条）
fig, ax = plt.subplots(figsize=(10,7))
x = np.arange(len(conditions))
bars = ax.bar(x, means, yerr=sems, capsize=5, color='skyblue', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel('Value')
ax.set_title('各条件下的数值及所有两两 T 检验显著性')

# 构造所有两两比较的列表，每项存储比较信息
pairs = []
n = len(conditions)
for i in range(n):
    for j in range(i+1, n):
        group1 = conditions[i]
        group2 = conditions[j]
        # 基准高度：取两组中 (均值 + 标准误) 的最大值
        base_y = max(means[group1] + sems[group1], means[group2] + sems[group2])
        # 进行 T 检验
        t_stat, p_val = stats.ttest_ind(df[group1], df[group2])
        sig_text = significance_marker(p_val)
        pairs.append({
            'i': i,
            'j': j,
            'group1': group1,
            'group2': group2,
            'base_y': base_y,
            'p_val': p_val,
            'sig_text': sig_text
        })

# 为避免标注线重叠，采用一个简单算法分配“层级”
# 先按基准高度从低到高排序
pairs.sort(key=lambda x: x['base_y'])
processed = []  # 存放已分配层级的比较

margin = 0.05  # 基准间隔
height = 0.05  # 每一层之间的间隔

for pair in pairs:
    level = 0
    # 检查与已处理的比较是否在水平区间上重叠且层级相同
    while True:
        conflict = False
        for other in processed:
            if other['level'] == level:
                # 如果两个比较的 x 区间有重叠，则认为冲突
                # 比较区间：[pair['i'], pair['j']] 与 [other['i'], other['j']]
                if not (pair['j'] < other['i'] or pair['i'] > other['j']):
                    conflict = True
                    break
        if conflict:
            level += 1
        else:
            break
    pair['level'] = level
    pair['y_line'] = pair['base_y'] + margin + level * height
    processed.append(pair)

# 在图中标注所有比较的显著性
line_h = 0.03  # 标注线的小高度
for pair in pairs:
    x1 = pair['i']
    x2 = pair['j']
    y_line = pair['y_line']
    # 绘制标注线：从 (x1, y_line) 到 (x1, y_line+line_h) 到 (x2, y_line+line_h) 再到 (x2, y_line)
    ax.plot([x1, x1, x2, x2], [y_line, y_line+line_h, y_line+line_h, y_line], lw=1.5, c='k')
    # 在中间添加文本，显示显著性符号和 p 值
    ax.text((x1+x2)*0.5, y_line+line_h, f"{pair['sig_text']}\np = {pair['p_val']:.3f}",
            ha='center', va='bottom', color='k', fontsize=8)

plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 定义显著性标注函数
def significance_marker(p_val):
    if p_val < 0.05:
        return "*"
    elif p_val < 0.1:
        return "°"
    else:
        return "n.s."

# 数据整理：每个条件包含 3 个样本
data = {
    'S0':   [0.840775251, 1.041348597, 1.267031855],
    'S2.5': [1.30079653,  1.284156476, 1.447891771],
    'S5':   [1.604879705, 1.32345248,  1.342409364],
    'S10':  [1.695382731, 1.944417514, 2.283764598],
    'S15':  [2.585801964, 2.474860641, 2.336889745]
}

df = pd.DataFrame(data)
conditions = list(data.keys())

# 计算每组的均值与标准误
means = df.mean()
sems = df.sem()

# 绘制柱状图（均值 + 误差条）
fig, ax = plt.subplots(figsize=(10,7))
x = np.arange(len(conditions))
bars = ax.bar(x, means, yerr=sems, capsize=5, color='skyblue', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel('Value')
ax.set_title('各条件下的数值及所有两两 T 检验显著性')

# 构造所有两两比较的列表，每项存储比较信息
pairs = []
n = len(conditions)
for i in range(n):
    for j in range(i+1, n):
        group1 = conditions[i]
        group2 = conditions[j]
        # 基准高度：取两组中 (均值 + 标准误) 的最大值
        base_y = max(means[group1] + sems[group1], means[group2] + sems[group2])
        # 进行 T 检验
        t_stat, p_val = stats.ttest_ind(df[group1], df[group2])
        sig_text = significance_marker(p_val)
        pairs.append({
            'i': i,
            'j': j,
            'group1': group1,
            'group2': group2,
            'base_y': base_y,
            'p_val': p_val,
            'sig_text': sig_text
        })

# 为避免标注线重叠，采用一个简单算法分配“层级”
# 先按基准高度从低到高排序
pairs.sort(key=lambda x: x['base_y'])
processed = []  # 存放已分配层级的比较

margin = 0.05  # 基准间隔
height = 0.05  # 每一层之间的间隔

for pair in pairs:
    level = 0
    # 检查与已处理的比较是否在水平区间上重叠且层级相同
    while True:
        conflict = False
        for other in processed:
            if other['level'] == level:
                # 如果两个比较的 x 区间有重叠，则认为冲突
                # 比较区间：[pair['i'], pair['j']] 与 [other['i'], other['j']]
                if not (pair['j'] < other['i'] or pair['i'] > other['j']):
                    conflict = True
                    break
        if conflict:
            level += 1
        else:
            break
    pair['level'] = level
    pair['y_line'] = pair['base_y'] + margin + level * height
    processed.append(pair)

# 在图中标注所有比较的显著性
line_h = 0.03  # 标注线的小高度
for pair in pairs:
    x1 = pair['i']
    x2 = pair['j']
    y_line = pair['y_line']
    # 绘制标注线：从 (x1, y_line) 到 (x1, y_line+line_h) 到 (x2, y_line+line_h) 再到 (x2, y_line)
    ax.plot([x1, x1, x2, x2], [y_line, y_line+line_h, y_line+line_h, y_line], lw=1.5, c='k')
    # 在中间添加文本，显示显著性符号和 p 值
    ax.text((x1+x2)*0.5, y_line+line_h, f"{pair['sig_text']}\np = {pair['p_val']:.3f}",
            ha='center', va='bottom', color='k', fontsize=8)

plt.tight_layout()
plt.show()
