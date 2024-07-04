accelerate launch --num_processes 2 --mixed_precision "fp16" \
  _train_whatever.py \
  --pretrained_model_name_or_path="stable-diffusion-xl-base-1.0" \
  --image_encoder_path="image_encoder" \
  --root_dir="/cfs/cfs-o4lnof4r/aigc_data_public/openpose_controlnet_train/laion_aesthetics/" \
  --valid_indices_file="valid_indices.pkl" \
  --one_pixel_indices_file="error_indices.txt" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-5 \
  --weight_decay=0.01 \
  --output_dir="experiments/ip-adapter" \
  --max_train_steps=500000 \
  --checkpointing_steps=50000 \
  --report_to="wandb"