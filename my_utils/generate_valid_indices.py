# generate_valid_indices.py

import os
import pickle
from tqdm import tqdm
import random

def generate_valid_and_invalid_indices(root_dir, output_file, invalid_sample_size=10):
    valid_indices = []
    all_invalid_indices = []
    
    for folder_idx in tqdm(range(64), desc="Processing folders"):
        folder_name = f"{folder_idx:05d}"
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_idx in range(10000):
                idx = folder_idx * 10000 + file_idx
                file_name = f"{idx:09d}"
                image_file = os.path.join(folder_path, f"{file_name}.jpg")
                text_file = os.path.join(folder_path, f"{file_name}.txt")
                if os.path.exists(image_file) and os.path.exists(text_file):
                    valid_indices.append(idx)
                else:
                    all_invalid_indices.append(idx)
    
    # 随机选择一些 invalid indices 作为样本
    invalid_indices_sample = random.sample(all_invalid_indices, min(invalid_sample_size, len(all_invalid_indices)))
    
    # 保存 valid indices
    with open(output_file, 'wb') as f:
        pickle.dump(valid_indices, f)
    
    # 保存 invalid indices 样本
    with open('invalid_indices_sample.pkl', 'wb') as f:
        pickle.dump(invalid_indices_sample, f)
    
    print(f"Valid indices saved to {output_file}")
    print(f"Total valid indices: {len(valid_indices)}")
    print(f"Sample of invalid indices saved to invalid_indices_sample.pkl")
    print(f"Total invalid indices: {len(all_invalid_indices)}")
    print("Sample of invalid indices:")
    for idx in invalid_indices_sample:
        folder_idx = idx // 10000
        file_idx = idx % 10000
        print(f"  {folder_idx:05d}/{idx:09d}.jpg")

if __name__ == "__main__":
    root_dir = "/cfs/cfs-o4lnof4r/aigc_data_public/openpose_controlnet_train/laion_aesthetics"  # 替换为您的数据集路径
    output_file = "valid_indices.pkl"
    generate_valid_and_invalid_indices(root_dir, output_file)
