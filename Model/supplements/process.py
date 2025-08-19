import numpy as np
import os


import pandas as pd
import numpy as np
import os

# file_path = '/home/yanghao/SummerRA/Code/ZSAR/My/supplements/hku.xlsx'
file_path = '/home/yanghao/SummerRA/Code/ZSAR/My/supplements/hku_test.xlsx'
# file_path = '/home/yanghao/SummerRA/Code/ZSAR/PURLS/supplements/gpt3_desc.xlsx'

df = pd.read_excel(file_path, sheet_name='Sheet1')
# df = pd.read_excel(file_path, sheet_name='NTU-RGB+D')

cls_labels=df.values[:,1:]

np.save("/home/yanghao/data/HKU/hku_skeleton_25/hku_bpnames.npy", cls_labels)
# np.save("/home/yanghao/data/synse_resources/resources/ntu60_bpnames.npy", cls_labels)


print(cls_labels.shape)
# print(cls_labels)
# print(cls_labels[:,1])

# cls_labels = np.load('/root/autodl-tmp/data/hku_skeletons/hku_bpnames.npy',allow_pickle=True)

# seen_inds=[1,2,3]
# unseen_inds=[4,5,6]

# seen_labels = [cls_labels[i] for i in seen_inds]
# unseen_labels = [cls_labels[i] for i in unseen_inds]

# print(seen_labels)
