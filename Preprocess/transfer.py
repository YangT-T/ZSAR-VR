import os
import shutil

# 输入和输出路径
input_dir = '/home/yanghao/SummerRA/Code/slicetest/final/data/output'
output_dir = '/home/yanghao/data/HKU/hku_skeleton_25/csv'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 定义 action 和 camera 映射字典
action_dict = {
    'walking': 1,
    'running': 2,
    'jumping': 3,
    'bending-down': 4,
    'stand': 5,
    'squatting': 6,
    'raising-hand': 7,

    'waive': 8,
    'throw': 9,
    'cut': 10,
    'shooting': 11,
    'bowling': 12,
    'move-using-controller': 13,
    'waive-sword': 14,
    'measure-length': 15,
    'picking-up-an-item-from-the-table-': 16,
    'throwing-a-net-to-catch-fish': 17,
    'grab-and-collect--box': 18
}

camera_dict = {
    'C': '001',
    'L': '002',
}

game_dict={
    'boss':1,
    'bowling':2,
    'candy':3,
    'gallery':4,
    'museum':5,
    'gaming museum':5,
    'travel':6
}



# 遍历 input_dir 下的所有 .csv 文件
cnt=0
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)

            # 从文件名中提取信息
            filename = file.replace('.csv', '')
            parts = filename.split('_')
            
            if len(parts)<6:
                continue

            # 示例: csv_01_boss_C_row1_rep1_Walking
            # 假设格式是固定的，按顺序提取
            try:
                view=camera_dict.get(parts[3]) 
                person=(3-len(parts[1]))*'0'+parts[1]
                action=action_dict.get(str.lower(parts[-1]))
                repetition=(6-len(parts[-2]))*'0'+parts[-2].replace('rep','')
                game=game_dict.get(parts[2])
                print(parts)
                row=int(parts[4].replace('row',''))
                
                if action is None or view is None :
                    continue
                
                new_name = f'S{game:03d}C{view}P{person}R{repetition}A{action:03d}L{row:03d}.csv'
                # 构造输出路径
                output_path = os.path.join(output_dir, new_name)

                # 复制并重命名文件
                shutil.copy(file_path, output_path)
                print(f"Renamed: {file} -> {new_name}")
                cnt+=1

            except StopIteration as e:
                print(f"Skipping invalid filename format: {file}")

print(cnt)