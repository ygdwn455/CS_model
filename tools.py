#!/usr/bin/env python3
# coding: utf-8

"""
auchor : XiaoY

工具包
preprocess_outlier : 异常值处理
"""

import numpy as np
import matplotlib.pyplot as plt


def preprocess_outlier(data_x, method='mean', whis=1.5):
    """
    异常值处理
    
    参数
    --------------
    data_x : 一维numpy数组
        要处理的数据
    method : str
        处理方式
    """
    data_tmp = data_x.ravel().astype(np.float).copy()
    outliers = plt.boxplot(data_tmp, whis=whis).get('fliers')[0].get_ydata()
    #print(outliers)
    inlier_mask = np.ones_like(data_tmp)
    for outlier in outliers:
        inlier_mask = np.logical_and(inlier_mask, data_tmp != outlier)
    #print(inlier_mask)
    outlier_mask = np.logical_not(inlier_mask)
    data_tmp[outlier_mask] = np.mean(data_tmp[inlier_mask])
    return data_tmp.reshape(data_x.shape)


def train_test_split(data, test_size=0.3):
    """
    平衡切分数据集
    
    参数
    --------------------------
    data : numpy数组
        输入集
    
    返回值
    --------------------------
    训练集分割得到的训练集和测试集
    """
    data_0 = data[data[:, 0]==0]
    data_0_test_len = int(len(data_0)*test_size)
    data_0_test = data_0[:data_0_test_len]
    data_0_train = data_0[data_0_test_len:]
    
    data_1 = data[data[:, 0]==1]
    data_1_test_len = int(len(data_1)*test_size)
    data_1_test = data_1[:data_1_test_len]
    data_1_train = data_1[data_1_test_len:]
    
    data_test = np.vstack((data_0_test, data_1_test))
    data_train = np.vstack((data_0_train, data_1_train))

    np.random.seed(0)
    data_test_randrom = data_test[np.random.permutation(range(0, len(data_test)))]
    data_train_randrom = data_train[np.random.permutation(range(0, len(data_train)))]
    
    return data_train_randrom, data_test_randrom

def calc_woe(data_y):
    """
    计算WOE的装饰器
    
    参数
    ------------------------
    data_y : 一维numpy数组
    
    返回值
    ------------------------
    woe : function
        真正计算WOE的函数
    """
    y_0_count, y_1_count = np.bincount(data_y.astype(np.int)).tolist()  # 计数numpy数组中各值的数量
    def woe(data_x, start, end):
        good, bad = np.bincount(data_y[np.logical_and(data_x>start, data_x<=end)].astype(np.int)).tolist()
        _woe = np.log((bad/y_1_count)/(good/y_0_count))
        return float(_woe)
    return woe

def calc_x_woe(data_x, y, range_list):
    """
    计算特征分箱的WOE
    
    参数
    -------------------------
    data_x : 一维numpy数组
    
    y : 一维numpy数组
    
    range_list : list
        特征取值范围
    
    返回值
    -------------------------
    train_woe : list
        特征取值范围对应WOE
    """
    train_woe = []
    start = -1
    end = -1
    woe = calc_woe(y)
    #print(range_list)
    for value in range_list[1:]:
        start = end
        end = value
        #print(start, end)
        woe_value = woe(data_x, start, end)
        train_woe.append([woe_value, [start, end]])
    return train_woe

def calc_score(woe_score, x, base_score):
    """
    计算评分
    
    参数
    ----------------
    woe_score : list
        分箱对应评分
    x : numpy数组(可改为多条数据的计算评分)
        输入数据
    base_score : float
        基础分
    
    返回值
    score_list : list
        评分
    """
    if len(x.shape) == 1: # 将一条数据转为二维, 便于处理
        X = np.array([x])
    elif len(x.shape) == 2:
        X = x.copy()
    else:
        print('输入数据有误。')
        return -1
    #X = X[:, [2,3,5,7,8,9,10]]
    score_list = []
    for row_index in X: # 计算每条数据
        #print(row_index)
        score = base_score
        for col_index, x in enumerate(row_index): # 从每条数据中循环使用特征数字
            #print('%d %f' % (col_index, x))
            for woe in woe_score[col_index]: # 进行判断每特征属于的范围，继而得到评分
                if x>woe[1][0] and x<=woe[1][1]:
                    score += woe[0]
                    #print(woe[0])
        score_list.append(float(score))
    return score_list

