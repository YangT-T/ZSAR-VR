import argparse
import pickle
from tqdm import tqdm
import sys
import csv

sys.path.extend(['../'])
from preprocess import pre_normalization

import numpy as np
import os


zaxis=[0,3]
xaxis=[1,2]



def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()
    else:
        s = 0
    return s


def read_csv_xyz(file_path, raw_num_joint=24):
    """
    读取 CSV 文件，返回 shape 为 (T, 25, 3) 的 numpy array
    自动填补缺失值为 0.0，并在末尾加一个 dummy joint
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        data = []
        for row in reader:
            if not row or len(row) < raw_num_joint * 3:
                print(f"存在无效或过短的行，跳过: {row}")
                continue
            try:
                float_row = [float(x) if x.strip() != '' else 0.0 for x in row]
                if len(float_row) != raw_num_joint * 3:
                    print(f"行长度不符，跳过: {row}")
                    continue
                data.append(float_row)
            except ValueError:
                print(f"存在无效行，跳过: {row}")
                continue

    data = np.array(data, dtype=np.float32)  # shape: (T, V*3)
    T = data.shape[0]
    data = data.reshape(T, raw_num_joint, 3)  # → (T, 24, 3)

    if data.shape[2]==24:
        # ➕ 添加 dummy joint：shape = (T, 1, 3)，→ concat → (T, 25, 3)
        zero_joint = np.zeros((T, 1, 3), dtype=np.float32)
        data = np.concatenate((data, zero_joint), axis=1)

    return data

def gendata_all_recursive(data_path, out_path, num_joint, max_file=0):
    sample_name = []
    sample_label = []
    data_list = []
    sample_filenames = []  # 新增：存文件名

    max_body_true = 2  # ✅ 固定为2人
    cnt = 0

    for root, dirs, files in os.walk(data_path):
        for filename in tqdm(files):
            if filename.endswith(".csv"):
                full_path = os.path.join(root, filename)
                try:
                    action_class = int(
                        filename[filename.find('A') + 1:filename.find('A') + 4])

                    data = read_csv_xyz(full_path, raw_num_joint=num_joint)
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                    T = data.shape[0]
                    C = 3

                    if T > max_frame:
                        data = data[:max_frame]
                        T = max_frame

                    data = data.transpose(2, 0, 1)

                    sample_array = np.zeros((C, max_frame, num_joint, max_body_true), dtype=np.float32)
                    sample_array[:, :T, :, 0] = data

                    sample_name.append(full_path)
                    sample_filenames.append(filename.split('.')[0])  # 记录文件名
                    sample_label.append(action_class - 1)
                    data_list.append(sample_array)

                    cnt += 1
                    if max_file > 0 and cnt >= max_file:
                        break

                except Exception as e:
                    print(f"读取失败 {full_path}: {e}")
    
    
    from collections import Counter
    counts = Counter(sample_label)

    # 打印统计结果
    for label, count in counts.items():
        print(f"{label}: {count}")


    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, 'all_label.pkl'), 'wb') as f:
        pickle.dump((sample_name, sample_label), f)

    with open(os.path.join(out_path, 'all_filenames.pkl'), 'wb') as f:
        pickle.dump(sample_filenames, f)

    fp = np.stack(data_list)
    fp = pre_normalization(fp, zaxis=zaxis, xaxis=xaxis)
    print(fp.shape)
    np.save(os.path.join(out_path, 'all_data_joint.npy'), fp)
    print(f"保存完毕: {len(sample_name)} 个样本")


def load_filenames(path):
    with open(path, 'rb') as f:
        filenames = pickle.load(f)
    return filenames


if __name__ == '__main__':
    data_path = "/home/yanghao/data/HKU/hku_skeleton_25/csv"
    out_dir = "/home/yanghao/data/HKU/hku_skeleton_25/processed"
    num_joint = 25  # ← 输出所需关节数
    max_body_true = 2
    max_frame = 300
    
    

    gendata_all_recursive(data_path, out_dir,num_joint,max_file=10000000)
    # gendata(data_path,out_path)
