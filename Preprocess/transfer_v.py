import os
import shutil

# 输入和输出路径
input_dir = '/home/yanghao/data/HKU/raw'
output_dir = '/home/yanghao/data/HKU/hku_skeleton_25/video'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 定义 action 和 camera 映射字典
action_dict = {
    'walking': 1,
    'running': 2,
    'jumping': 3,

    'bending-down': 4,
    'bending down': 4,

    'stand': 5,
    'squatting': 6,

    'raising-hand': 7,
    'raising hand': 7,


    'waive': 8,
    'throw': 9,
    'cut': 10,
    'shooting': 11,
    'bowling': 12,
    'move-using-controller': 13,
    'move using controller': 13,
    'waive-sword': 14,
    'waive sword': 14,
    'measure-length': 15,
    'measure length': 15,
    'picking-up-an-item-from-the-table-': 16,
    'picking up an item from the table': 16,
    'picking up an item from the table ': 16,
    'throwing-a-net-to-catch-fish': 17,
    'throwing a net to catch fish': 17,
    'grab-and-collect--box': 18,
    'grab and collect box': 18,
    'grab and collect  box': 18
}


game_dict={
    'boss':1,
    'bowling':2,
    'candy':3,
    'gallery':4,
    'museum':5,
    'travel':6
}

camera_dict = {
    'C': '001',
    'L': '002',
}


# 遍历 input_dir 下的所有 .mp4 文件
cnt=0
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.mp4'):
            file_path = os.path.join(root, file)

            # 从文件名中提取信息
            filename = file.replace('.mp4', '')
            parts = filename.split('_')
            
            if len(parts)<6:
                continue

            # 示例: 01_boss_C_cut_row24_rep1
            # 假设格式是固定的，按顺序提取
            try:
                view=camera_dict.get(parts[2]) 
                person=(3-len(parts[0]))*'0'+parts[0]
                action=action_dict.get(str.lower(parts[3]))
                repetition=(6-len(parts[-1]))*'0'+parts[-1].replace('rep','')
                game=game_dict.get(parts[1])
                print(parts)
                row=int(parts[4].replace('row',''))
   
                
                if action is None or view is None :
                    print(f'fuck:{str.lower(parts[3])}')
                    continue
                
                new_name = f'S{game:03d}C{view}P{person}R{repetition}A{action:03d}L{row:03d}.mp4'
                # 构造输出路径
                output_path = os.path.join(output_dir, new_name)

                # 复制并重命名文件
                shutil.copy(file_path, output_path)
                print(f"Renamed: {file} -> {new_name}")
                cnt+=1

            except StopIteration as e:
                print(f"Skipping invalid filename format: {file}")

print(cnt)