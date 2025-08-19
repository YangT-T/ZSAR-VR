import cv2
import os
import pandas as pd

# --- CONFIGURATION ---
video_root = '/data/sda1/mocap_data/raw_video'  # A path that contains all videos
output_path='/home/yanghao/data/HKU/raw/'       # Sliced output path (format aligns with csv files)

video_format = 'mp4'               

def slice(video_code,video_suffix,video_folder):
    sheet_video_map = {
        'Gaming Museum':   f'{video_code}_museum_{video_suffix}.mp4',
        'BowlingVR':       f'{video_code}_bowling_{video_suffix}.mp4',
        'Gallery of H.K. History': f'{video_code}_gallery_{video_suffix}.mp4',
        'Hong Kong Time Travel':   f'{video_code}_travel_{video_suffix}.mp4',
        'Boss Fight':      f'{video_code}_boss_{video_suffix}.mp4',
        'Candy Shooter':   f'{video_code}_candy_{video_suffix}.mp4'
    }
    # --- SETUP ---
    for sheet_name, video_file in sheet_video_map.items():
        video_path = os.path.join(video_folder, video_file)
        game = video_file.split('_')[1]
        print(video_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')


        
        output_folder = os.path.join(os.path.dirname(output_path), f"{video_code}_{game}")
        os.makedirs(output_folder, exist_ok=True)
        excel_path = f'/data/sda1/excel/DataCollection_{video_code}.xlsx'
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        rep_start_cols = [col for col in df.columns if 'Repetition' in col and 'Start' in col]
        rep_end_cols = [col for col in df.columns if 'Repetition' in col and 'End' in col]
        rep_nums = set()
        for col in rep_start_cols + rep_end_cols:
            try:
                rep_num = int(col.split()[1])
                rep_nums.add(rep_num)
            except:
                continue
        num_repetitions = max(rep_nums) if rep_nums else 0
        print(f"[{sheet_name}] max repetitions = {num_repetitions}")


        action=''
        for row_idx, row in df.iterrows():
            for rep in range(1, num_repetitions + 1):
                start_col = f"Repetition {rep} Start"
                end_col = f"Repetition {rep} End"

                start = row.get(start_col)
                end = row.get(end_col)

                temp = row.get('Action')
                if not pd.isna(temp):
                    action= temp

                if pd.notna(start) and pd.notna(end):
                    start_frame = int(start)
                    end_frame = int(end)
                    clip_filename = os.path.join(
                        output_folder,
                        f"{video_code}_{game}_{video_suffix}_{action}_row{row_idx+1}_rep{rep}.{video_format}"
                    )

                    out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                    for i in range(start_frame, end_frame + 1):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)

                    out.release()
                    print(f"Saved: {clip_filename}")
                # else:
                #     print(f"Skipped: row {row_idx+1}, repetition {rep} (missing data)")

        cap.release()

for video_suffix in ['C','L']:
    for id in range(15):
        id+=1
        if id<=5:
            continue

        video_code = f'{id:02d}'
        
        matched_folders = []

        for folder_name in os.listdir(video_root):
            folder_path = os.path.join(video_root, folder_name)
            if os.path.isdir(folder_path) and folder_name.endswith(video_code):
                slice(video_code,video_suffix,folder_path)
                print(f'slicing {folder_path}')
                matched_folders.append(folder_name)

        # print("匹配到的文件夹：", matched_folders)


print("All available clips saved.")
