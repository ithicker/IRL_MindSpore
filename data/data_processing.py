import pandas as pd
import numpy as np
import random
import codecs
from collections import defaultdict
import numpy as np

def gen_uniform_interval(target, maximum_interval, left, right):
    """Generating Weakly Supervised Information via Uniform Distribution"""
    a = np.random.uniform(left, right)
    b = np.random.uniform(left, right)
    while ((b - a) > maximum_interval) or (a > target) or (b < target):
        a = np.random.uniform(left, right)
        b = np.random.uniform(left, right)
    interval = np.array([a, b])
    return interval
def check_interval(interval, left, right):
    if len(interval.shape) == 2:
        for i in interval:
            if i[0] < left:
                i[0] = left
            if i[1] > right:
                i[1] = right
    elif len(interval.shape) == 1:
        if interval[0] < left:
            interval[0] = left
        if interval[1] > right:
            interval[1] = right


def cut_list(l, ratio):
    random.shuffle(l)
    length = len(l)
    idx = random.sample(range(length), int(length * ratio))
    l1 = []
    l2 = []
    index = [False for _ in range(length)]
    for i in idx:
        index[i] = True
    for i, j in enumerate(index):
        if j:
            l1.append(l[i])
        else:
            l2.append(l[i])
    return l1, l2


def slicing(df, ratio):
    df = df.reset_index(drop=True)
    l = [i for i in range(len(df))]
    _, l2 = cut_list(l, ratio)
    for i in l2:
        df["split"][i] = "-"
    return df.loc[df['split'] == "train"]



def AgeDB_data_processing(max_interval, left, right, path_load='data/dataset/agedb.csv',
                          path_save="data/dataset/agedb_split"):
    df = pd.read_csv(path_load)
    label = np.asarray(df['age']).astype('float32')

    interval = [gen_uniform_interval(x, max_interval, left, right) for x in label]
    interval = np.array(interval)
    check_interval(interval, 0, 200)
    df["yl"] = interval[:, 0]
    df["yr"] = interval[:, 1]
    l = [i for i in range(len(df))]
    l1, l2 = cut_list(l, 0.6)
    l2, l3 = cut_list(l2, 0.5)
    df["split"] = "train"
    for i in l2:
        df["split"][i] = "verify"
    for i in l3:
        df["split"][i] = "test"
    path_save = path_save + "/agedb_" + str(max_interval) + ".csv"
    df.to_csv(path_save, sep=",")





