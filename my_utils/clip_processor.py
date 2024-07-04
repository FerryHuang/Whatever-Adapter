from transformers.models.clip import CLIPImageProcessor
import pickle
import os

clip_processor = CLIPImageProcessor()
error_indices_file = "error_indices.txt"
dataset_path = "/cfs/cfs-o4lnof4r/aigc_data_public/openpose_controlnet_train/laion_aesthetics/"
valid_indices_file = "valid_indices.pkl"
with open(valid_indices_file, 'rb') as f:
    valid_indices = pickle.load(f)
print("all images number:", len(valid_indices))

# 加载 invalid_indices
one_pixel_indices_file = "error_indices.txt"
if os.path.exists(one_pixel_indices_file):
    with open(one_pixel_indices_file, 'r') as f:
        invalid_indices = set([int(line.strip()) for line in f])
    
    # 从 valid_indices 中移除 invalid_indices 指定位置的元素
    valid_indices = [idx for i, idx in enumerate(valid_indices) if i not in invalid_indices]

print("valid images number:", len(valid_indices))