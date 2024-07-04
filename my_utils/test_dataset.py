import time
from train_whatever_adapter import MyDataset
from transformers.models.clip import CLIPTokenizer
from torch.utils.data import DataLoader
import torch

pretrained_model_name_or_path = "stable-diffusion-xl-base-1.0"
dataset_path = "/cfs/cfs-o4lnof4r/aigc_data_public/openpose_controlnet_train/laion_aesthetics/"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(filename='dataset_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def test_dataset(dataset_path, valid_indices_file, batch_size=1, num_workers=4):
    # 初始化数据集
    dataset = MyDataset(root_dir=dataset_path, valid_indices_file=valid_indices_file,
                        tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=1024)
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers)
    
    error_indices = []
    
    # 直接遍历数据集
    for idx in tqdm(range(len(dataset)), desc="Testing dataset"):
        try:
            # 获取单个项目
            item = dataset[idx]
            
            # 访问项目中的数据以触发实际的数据加载
            _ = item["image"]
            _ = item["text_input_ids"]
            _ = item["clip_image"]
        except Exception as e:
            # 记录错误
            error_message = f"Error in item {idx}: {str(e)}"
            logging.error(error_message)
            error_indices.append(idx)
    
    print(f"Total errors: {len(error_indices)}")
    print(f"Error indices: {error_indices}")
    
    # 将错误索引保存到文件
    with open('error_indices.txt', 'w') as f:
        for index in error_indices:
            f.write(f"{index}\n")

if __name__ == "__main__":
    dataset_path = "/cfs/cfs-o4lnof4r/aigc_data_public/openpose_controlnet_train/laion_aesthetics/"
    valid_indices_file = "valid_indices.pkl"
    test_dataset(dataset_path, valid_indices_file)
