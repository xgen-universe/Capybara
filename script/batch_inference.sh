# ============== Configuration ==============
num_processes=4
task="tv2v"
resolution="480p"  # "480p", "720p", "1024", "1080p"
frames=81          # 81, 101, 121

# Paths (modify these to your own paths)
pretrained_model_path="./ckpts"
output_path="./results/$task-$resolution"
csv_path="./assets/test_data/tv2v_example.csv"
data_root_path="./assets/examples"


# ============== Launch ==============
# For single GPU (no distributed):
python inference.py \
    --pretrained_model_name_or_path "$pretrained_model_path" \
    --csv_path "$csv_path" \
    --output_path "$output_path" \
    --data_root_path "$data_root_path" \
    --num_inference_steps 50 \
    --num_frames $frames \
    --task_type $task \
    --resolution $resolution \
    --rewrite_instruction

# For multi-GPU (distributed), use:
# accelerate launch --config_file acc_config/accelerate_config.yaml --num_processes $num_processes inference.py \
#     --pretrained_model_name_or_path "$pretrained_model_path" \
#     --csv_path "$csv_path" \
#     --output_path "$output_path" \
#     --data_root_path "$data_root_path" \
#     --num_inference_steps 50 \
#     --num_frames $frames \
#     --task_type $task \
#     --resolution $resolution \
#     --rewrite_instruction
