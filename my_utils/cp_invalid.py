import os
import shutil
import pickle
from tqdm import tqdm

def copy_images(dataset_path, output_folder, error_indices_file, valid_indices_file):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取 valid_indices
    with open(valid_indices_file, 'rb') as f:
        valid_indices = pickle.load(f)
    
    # 读取错误索引
    with open(error_indices_file, 'r') as f:
        error_indices = [int(line.strip()) for line in f]
    
    # 复制图片
    for idx in tqdm(error_indices, desc="Copying images"):
        if idx < len(valid_indices):
            actual_idx = valid_indices[idx]
            folder_idx = actual_idx // 10000
            file_idx = actual_idx % 10000
            folder_name = f"{folder_idx:05d}"
            file_name = f"{actual_idx:09d}.jpg"
            
            source_path = os.path.join(dataset_path, folder_name, file_name)
            dest_path = os.path.join(output_folder, file_name)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
            else:
                print(f"File not found: {source_path}")
        else:
            print(f"Index {idx} is out of range for valid_indices")

if __name__ == "__main__":
    dataset_path = "/cfs/cfs-o4lnof4r/aigc_data_public/openpose_controlnet_train/laion_aesthetics/"  # 替换为您的数据集路径
    output_folder = "processor_error_imgs"  # 替换为您想要保存图片的文件夹路径
    error_indices_file = "error_indices.txt"  # 包含错误索引的文件
    valid_indices_file = "valid_indices.pkl"  # 包含有效索引的文件
    
    copy_images(dataset_path, output_folder, error_indices_file, valid_indices_file)
