# ============== Configuration ==============
task="ti2i"
resolution="720p"  # "480p", "720p", "1024", "1080p"
frames=81          # 81, 101, 121

# Paths (modify these to your own paths)
pretrained_model_path="./ckpts"
output_path="./results/$task-$resolution"

media_path="./assets/examples/img1.jpeg"
prompt="Change the time to night."

# ============== Single Sample Mode ==============
# For single image editing (ti2i):
python inference.py \
    --pretrained_model_name_or_path "$pretrained_model_path" \
    --media_path "$media_path" \
    --prompt "$prompt" \
    --output_path "$output_path" \
    --num_inference_steps 50 \
    --num_frames $frames \
    --task_type $task \
    --resolution $resolution \
    --rewrite_instruction

# For single video editing (tv2v):
# python inference.py \
#     --pretrained_model_name_or_path "$pretrained_model_path" \
#     --media_path "/path/to/your/video.mp4" \
#     --prompt "Apply a ghibli style to this video" \
#     --output_path "./results/test_single_output/tv2v" \
#     --num_inference_steps 50 \
#     --num_frames 81 \
#     --task_type tv2v \
#     --resolution 720p \
#     --rewrite_instruction

# For text-to-video generation (t2v, no media_path needed):
# python inference.py \
#     --pretrained_model_name_or_path "$pretrained_model_path" \
#     --prompt "A giant humpback whale and its calf gracefully swim in the crystal-clear, deep blue open ocean." \
#     --output_path "./results/test_single_output/t2v" \
#     --guidance_scale 4.0 \
#     --num_inference_steps 50 \
#     --num_frames 81 \
#     --task_type t2v \
#     --resolution 720p \
#     --aspect_ratio "16:9" \
#     --rewrite_instruction

# For text-to-image generation (t2i, no media_path needed):
# python inference.py \
#     --pretrained_model_name_or_path "$pretrained_model_path" \
#     --prompt "A group of five hikers, sitting on the snow mountain." \
#     --output_path "./results/test_single_output/t2i" \
#     --guidance_scale 4.0 \
#     --num_inference_steps 50 \
#     --task_type t2i \
#     --resolution 720p \
#     --aspect_ratio "16:9" \
#     --rewrite_instruction
