#!/usr/bin/env python3
# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import RandomizedLogisticRegression

import tools

"""
信用模型分析（Python）
"""

"""
1.数据准备
> 使用pandas读取所要分析的数据
"""
cs_train = pd.read_csv("./data/cs-training.csv", index_col=0, header=0)
# 输出数据信息
print("*" * 20 + "   准备数据   " + "*" * 20)
print(cs_train.shape)          # 数据结构
columns = list(cs_train.columns)    # 数据特征列名
print(cs_train.columns)        # 数据特征名
print(cs_train.index)          # 数据索引
print("*" * 50)

"""
2.数据分析处理
1> 缺失值分析
"""
print("*" * 20 + "   缺失值分析   " + "*" * 20)
#figure_null_value = plt.figure()
#axes_null_value = figure_null_value.add_subplot(111)
for column in columns:
    rows = cs_train[cs_train.loc[:,column].isnull()].loc[:, column]
    if not rows.empty:
        print("特征 %s 存在 %d 缺失值" % (column, len(rows)))
print("*" * 50)
# 缺失值处理
cs_train_no_nan = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0).fit_transform(cs_train)

"""
2> 异常值分析
使用箱线图查看数据密集
"""
print("*" * 20 + "   异常值分析   " + "*" * 20)

for col_index in range(0, len(columns)):
    print("特征 %s 箱线图" % columns[col_index])
    figure_outlier_value = plt.figure()  #创建一个画布
    axes_outlier_value = figure_outlier_value.add_subplot(111)#创建一个坐标系(1,1,1)
    axes_outlier_value.boxplot(cs_train_no_nan[:,col_index])# 绘制箱线图
    plt.show()

#cs_train_not_nan[:, 3] = tools.preprocess_outlier(cs_train_not_nan[:, 3])
#cs_train_not_nan[:, 7] = tools.preprocess_outlier(cs_train_not_nan[:, 7])
#cs_train_not_nan[:, 9] = tools.preprocess_outlier(cs_train_not_nan[:, 9])
plt.show()

# 数据异常值删除处理
#(age 借款人当时的年龄)不为零
cs_train_tmp = cs_train_no_nan[cs_train_no_nan[:, 2] != 0]
print(cs_train_tmp.shape)
#(NumberOfTime30-59DaysPastDueNotWorse 35-59天逾期但不糟糕次数)小于95
cs_train_tmp = cs_train_tmp[cs_train_tmp[:, 3] < 95]
print(cs_train_tmp.shape)
#(NumberOfTimes90DaysLate 90天逾期次数)小于95
cs_train_tmp = cs_train_tmp[cs_train_tmp[:, 7] < 95]
print(cs_train_tmp.shape)
#(NumberOfTime60-89DaysPastDueNotWorse 60-89天逾期但不糟糕次数)小于95
cs_train_tmp = cs_train_tmp[cs_train_tmp[:, 9] < 95]
print(cs_train_tmp.shape)
cs_train_no_nanoutlier = cs_train_tmp
print("*" * 50)

"""
3> 特征分析
使用直方图查看数据特征
"""
for col_index in range(0, len(columns)):
    print("特征 %s 直方图" % columns[col_index])
    plt.hist(cs_train_no_nanoutlier[:, col_index])
    plt.show()

#plt.hist(cs_train_no_nanoutlier[:, 5], range=(1,20000))
#plt.xlim(1,20000)
#plt.show()

"""
# 绘制特征间散点图
seaborn.pairplot(cs_train_no_nanoutlier)
plt.tight_layout()
# plt.savefig('./figures/scatter.png', dpi=300)
plt.show()
"""

# 绘制皮尔逊相关系数热力图
cm = np.corrcoef(cs_train_no_nanoutlier.T)
seaborn.set(font_scale=1.5)
print(columns)
print(cm.shape)
hm = seaborn.heatmap(cm, 
                     cbar=True, 
                     annot=True, 
                     square=True, 
                     fmt='.2f', 
                     annot_kws={'size': 8}, 
                     yticklabels=columns, 
                     xticklabels=columns)

# plt.tight_layout()
plt.savefig('相关性矩阵.png', dpi=300)
plt.show()

"""
4> 切分数据集
由于输出数据中0与1数量相差较大，所以使用tools模块的自定义函数train_test_split做平衡分割
"""
cs_train_train, cs_train_test = tools.train_test_split(cs_train_no_nanoutlier, test_size=0.5)

"""
5> 选择特征（自行写出代码）
可根据逻辑回归权重结合实际选择
根据逻辑回归得出特征1,4,5,6权重较小，
但5主观上与结果较为相关，
所以删除特征1,4,6 
"""
cs_train_X = cs_train_train[:, [2,3,5,7,8,9,10]] # 选择的训练特征
cs_train_y = cs_train_train[:, 0] # 训练输出
# 测试集
cs_test_X = cs_train_test[:, [2,3,5,7,8,9,10]] 
cs_test_y = cs_train_test[:, 0]

# 建立逻辑回归并训练（自行加入验证与网格搜索等操作）
lr = LogisticRegression(penalty='l2')
lr.fit(cs_train_X, cs_train_y)

# predict_proba返回 每行数据样本我不同结果的概率
cs_test_y_proba = lr.predict_proba(cs_test_X)

"""
6> 绘制ROC曲线，观察拟合度
roc_curve : 绘制roc曲线
auc : 计算曲线下面积
"""
fpr, tpr, thresholds = roc_curve(cs_test_y, cs_test_y_proba[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label='预测roc曲线 %f' % roc_auc)
plt.plot([0, 1], [0, 1], '--r', label='随机roc曲线')
plt.plot([0, 0, 1], [0, 1, 1], 'r', label='完美roc曲线')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.show()
#cs_test_y_pred = lr.predict(cs_test_X)
#confmat = confusion_matrix(cs_test_y, cs_test_y_pred)
#print(confmat)
"""
[[41634   321]
 [ 2525   438]]
"""

"""
7> 计算所有特征不同分箱的WOE
"""
train_woe = [] # 存储所有WOE
train_woe.append(tools.calc_x_woe(cs_train_X[:, 0], cs_train_y, [-1, 30, 35, 40, 45, 50, 55, 65, 70, 75, 1000]))
train_woe.append(tools.calc_x_woe(cs_train_X[:, 1], cs_train_y, [-1, 0, 1, 3, 5, 1000]))
train_woe.append(tools.calc_x_woe(cs_train_X[:, 2], cs_train_y, [-1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,11000,12000,10000000]))
train_woe.append(tools.calc_x_woe(cs_train_X[:, 3], cs_train_y, [-1, 0, 1, 3, 5, 10, 1000]))
train_woe.append(tools.calc_x_woe(cs_train_X[:, 4], cs_train_y, [-1, 0, 1, 2, 3, 5, 1000]))
train_woe.append(tools.calc_x_woe(cs_train_X[:, 5], cs_train_y, [-1, 0, 1, 3, 5, 1000]))
train_woe.append(tools.calc_x_woe(cs_train_X[:, 6], cs_train_y, [-1, 0, 1, 2, 3, 5, 1000]))

"""
8> 将数据集转换为WOE
"""
cs_train_X_woe = np.zeros_like(cs_train_X)
for idx, woes in enumerate(train_woe):
    for woe in woes:
        tmp = cs_train_X[:, idx]
        cs_train_X_woe[np.logical_and(tmp > woe[1][0], tmp <= woe[1][1]), idx] = woe[0]
#cs_train_y_reverse = 1 - cs_train_y

"""
9> 构建逻辑回归，使用WOE数据集作为训练的输入
"""
lr_woe = LogisticRegression()
lr_woe.fit(cs_train_X_woe, cs_train_y)
coef = lr_woe.coef_ #　权重
intercept = lr.intercept_ # 截距

"""
10> 构建评分卡，score = A-B(odds)
"""
p = 20/np.log(2)
q = 600-20*np.log(15)/np.log(2)

base_score = q + p * intercept # 基础分

# 计算各特征的所有分箱的评分
train_woe_score = []
for idx, woes in enumerate(train_woe):
    tmp = []
    for woe in woes:
        tmp.append([int(woe[0]*p*coef[0, idx]), woe[1]])
    train_woe_score.append(tmp)

scores = tools.calc_score(train_woe_score, cs_test_X[:5], base_score)
print(scores)

      