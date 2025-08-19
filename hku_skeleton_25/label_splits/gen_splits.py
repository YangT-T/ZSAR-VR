import numpy as np

# cls_num=7
# us_num=3

label_spilt_dir='/home/yanghao/data/HKU/hku_skeleton_25/label_splits'

# output_dir=f'/root/autodl-tmp/data/zsl_features/shift_{us_num}_r'


unseen_class=6

# if unseen_class==6:
#     s_label=[0,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17]
#     us_label=[1,11]
#     v_label=[0,3,5,12,17]
# if unseen_class==6:
#     s_label=[0,1,2,3,4,5,6,7,8,9,10,11,13,14,17]
#     us_label=[12,15,16]
#     v_label=s_label[0:3]


us_label = [3,6,15,16,17]
all_labels = list(range(18))  # 0~17，共18类
s_label = [x for x in all_labels if x not in us_label]
v_label = s_label[0:-1:len(s_label)//3]

print(us_label)
print(s_label)
print(v_label)

# s_label=np.load(label_spilt_dir+f'/rs.npy')
# v_label=np.load(label_spilt_dir+f'/rv.npy')
# us_label=np.load(label_spilt_dir+f'/ru.npy')

np.save(label_spilt_dir+f'/rs.npy',s_label)
np.save(label_spilt_dir+f'/rv.npy',v_label)
np.save(label_spilt_dir+f'/ru.npy',us_label)


