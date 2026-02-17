import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from diffusers.utils import export_to_video, load_image
from moviepy import VideoFileClip, concatenate_videoclips, clips_array
from accelerate import Accelerator
import imageio
import einops
import random
from PIL import Image
from safetensors.torch import load_file

from models.pipelines.capybara_pipeline import Capybara_Pipeline
from utils import (
    load_rewrite_model, 
    get_media_path_for_rewrite, 
    rewrite_instruction_fn,
    get_resolution,
    get_extra_kwargs,
    get_extra_kwargs_single
)


def parse_args():
    parser = argparse.ArgumentParser(description="Video Edit Inference Script")
    
    # Model paths
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to the first transformer model"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to the base pipeline model"
    )
    
    # Data path (CSV mode)
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to the CSV file containing video information"
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default=None,
        help="Root path for media files referenced in the CSV"
    )
    
    # Single sample mode
    parser.add_argument(
        "--media_path",
        type=str,
        default=None,
        help="Path to a single image or video file (single sample mode)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text for single sample mode"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames to generate (default: 49)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed number for inference"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for generation (default: 4.0)"
    )
    parser.add_argument(
        "--guidance_scale_2",
        type=float,
        default=1.0,
        help="Second guidance scale for generation (default: 3.0)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 40)"
    )
    
    # Output path
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Output directory path (default: ./output)"
    )
    parser.add_argument(
        "--num_sample_per_case",
        type=int,
        default=1,
        help="Number of samples per test case"
    )
    
    # Optional parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (default: cuda:0)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model (default: bfloat16)"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for generation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="tv2v",
        help="Task type"
    )
    parser.add_argument(
        "--key_frame_num",
        type=int,
        default=0,
        help="Key frame number for propagation task"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        choices=["480p", "720p", "1024", "1080p"],
        help="Resolution for t2v or ti2i tasks (480p=group1, 720p=group2, 1024=group3, 1080p=group4)"
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default=None,
        choices=["16:9", "9:16", "4:3", "3:4", "1:1"],
        help="Aspect ratio for t2v or ti2i tasks"
    )
    
    # Instruction rewrite parameters
    parser.add_argument(
        "--rewrite_instruction",
        action="store_true",
        default=False,
        help="Whether to rewrite/expand the editing instruction using Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--rewrite_model_path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Path or HuggingFace model name for Qwen3-VL-8B-Instruct (default: Qwen/Qwen3-VL-8B-Instruct)"
    )
    
    return parser.parse_args()


def save_video(video, path):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=24)


def save_image(video, path, frame_idx=0):
    """
    Save a single frame from a video tensor as a PNG image.
    
    Args:
        video: Video tensor with shape (B, C, F, H, W) or (C, F, H, W) or (C, H, W)
        path: Output path for the image
        frame_idx: Index of the frame to extract (default: 0, first frame)
    """
    # Handle 5D tensor (batch dimension)
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    
    # Handle 4D tensor (C, F, H, W) - extract frame
    if video.ndim == 4:
        # Extract the specified frame
        num_frames = video.shape[1]
        frame_idx = min(frame_idx, num_frames - 1)
        video = video[:, frame_idx, :, :]  # (C, H, W)
    
    # Convert to uint8 and rearrange to (H, W, C)
    img = (video * 255).clamp(0, 255).to(torch.uint8)
    img = einops.rearrange(img, 'c h w -> h w c')
    
    # Convert to numpy and save using PIL
    img_np = img.cpu().numpy()
    Image.fromarray(img_np).save(path)


def distribute_data(data, accelerator):
    """Distribute data across processes"""
    total_samples = len(data)
    per_process = total_samples // accelerator.num_processes
    remainder = total_samples % accelerator.num_processes
    
    # Calculate start and end indices for current process
    start_idx = accelerator.process_index * per_process
    if accelerator.process_index < remainder:
        start_idx += accelerator.process_index
        end_idx = start_idx + per_process + 1
    else:
        start_idx += remainder
        end_idx = start_idx + per_process
    
    # Get data slice for current process
    process_data = data[start_idx:end_idx]
    
    if accelerator.is_main_process:
        print(f"Total samples: {total_samples}")
        print(f"Distributed across {accelerator.num_processes} processes")
    
    print(f"Process {accelerator.process_index}: {len(process_data)} samples (indices {start_idx}-{end_idx-1})")
    
    return process_data, start_idx


def run_single_sample(args, pipe, accelerator, rewrite_model=None, rewrite_processor=None):
    """Run inference on a single image or video sample."""
    instruction = args.prompt
    extra_kwargs = get_extra_kwargs_single(args)

    # Rewrite instruction if enabled
    if args.rewrite_instruction and rewrite_model is not None:
        media_path = get_media_path_for_rewrite(extra_kwargs)
        instruction = rewrite_instruction_fn(
            instruction, media_path, rewrite_model, rewrite_processor
        )

    # get_resolution needs a row-like object; build a minimal one
    dummy_row = {}
    bucket, aspect_ratio = get_resolution(args, dummy_row, extra_kwargs)

    # Determine video length
    if args.task_type in ["ti2i", "t2i", "ii2i"]:
        video_length = 1
    else:
        video_length = args.num_frames

    print(f"Instruction : {instruction}")
    print(f"Media path  : {args.media_path}")
    print(f"Task type   : {args.task_type}")
    print(f"Resolution  : {bucket}, aspect_ratio: {aspect_ratio}")
    print(f"Video length: {video_length}")

    for j in range(args.num_sample_per_case):
        if args.seed is None:
            seed = random.randint(0, 2**30)
        else:
            seed = args.seed

        out = pipe(
            enable_sr=False,
            prompt=instruction,
            aspect_ratio=aspect_ratio,
            num_inference_steps=args.num_inference_steps,
            video_length=video_length,
            negative_prompt=args.negative_prompt,
            seed=seed,
            output_type="pt",
            prompt_rewrite=False,
            return_pre_sr_video=False,
            guidance_scale=args.guidance_scale,
            task_type=args.task_type,
            height=bucket[0],
            width=bucket[1],
            target_device=accelerator.device,
            **extra_kwargs,
        )

        suffix = f"_{j}" if args.num_sample_per_case > 1 else ""
        if args.task_type in ["ti2i", "t2i"]:
            out_file = os.path.join(args.output_path, f"output{suffix}.png")
            save_image(out.videos, out_file)
        else:
            out_file = os.path.join(args.output_path, f"output{suffix}.mp4")
            save_video(out.videos, out_file)
        print(f"Saved: {out_file}")


def run_csv_batch(args, pipe, accelerator, rewrite_model=None, rewrite_processor=None):
    """Run inference on a batch of samples from a CSV file."""
    # Load CSV data
    print(f"Loading CSV from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    if args.max_samples:
        df = df[:min(len(df), args.max_samples)]
    print(f"Processing {len(df)} video pairs")

    # Distribute data across processes
    accelerator.wait_for_everyone()
    process_data, process_start_idx = distribute_data(df, accelerator)
    accelerator.wait_for_everyone()

    # Process each row
    for idx, (i, row) in enumerate(process_data.iterrows()):
        global_idx = process_start_idx + idx
        
        if "editing_instruction" in row:
            instruction = row['editing_instruction']
        elif "instruction" in row:
            instruction = row['instruction']
        elif "caption" in row:
            instruction = row['caption']
        elif "prompt" in row:
            instruction = row['prompt']
        else:
            raise ValueError(f"No instruction found in row: {row}")

        extra_kwargs = get_extra_kwargs(args, row)

        # Rewrite instruction if enabled
        if args.rewrite_instruction and rewrite_model is not None:
            media_path = get_media_path_for_rewrite(extra_kwargs)
            instruction = rewrite_instruction_fn(
                instruction, media_path, rewrite_model, rewrite_processor
            )

        bucket, aspect_ratio = get_resolution(args, row, extra_kwargs)

        # For ti2i/t2i/ii2i tasks, set num_frames to 1
        if args.task_type in ["ti2i", "t2i", "ii2i"]:
            video_length = 1
        else:
            # If CSV provides a 'frames' column, use the smaller of num_frames and frames
            if "frames" in row and pd.notna(row["frames"]):
                video_length = min(args.num_frames, int(row["frames"]))
            else:
                video_length = args.num_frames

        for j in range(args.num_sample_per_case):
            if args.seed is None:
                seed = random.randint(0, 2**30)
            else:
                seed = args.seed

            # Forward instruction
            out = pipe(
                enable_sr=False,
                prompt=instruction,
                aspect_ratio=aspect_ratio,
                num_inference_steps=args.num_inference_steps,
                video_length=video_length,
                negative_prompt=args.negative_prompt,
                seed=seed,
                output_type="pt",
                prompt_rewrite=False,
                return_pre_sr_video=False,
                guidance_scale=args.guidance_scale,
                task_type=args.task_type,
                height=bucket[0],
                width=bucket[1],
                target_device=accelerator.device,
                **extra_kwargs,
            )
            
            if args.task_type in ["ti2i", "t2i"]:
                save_image(out.videos, os.path.join(args.output_path, f"{global_idx:02d}.png"))
            else:
                save_video(out.videos, os.path.join(args.output_path, f"{global_idx:02d}.mp4"))


def main():
    args = parse_args()

    # Validate input mode: either single sample or CSV batch
    single_sample_mode = args.media_path is not None or args.prompt is not None
    csv_mode = args.csv_path is not None

    if single_sample_mode and csv_mode:
        raise ValueError(
            "Cannot use both single sample mode (--media_path / --prompt) "
            "and CSV mode (--csv_path) at the same time."
        )

    if single_sample_mode:
        if args.prompt is None:
            raise ValueError("--prompt is required in single sample mode.")
        if args.media_path is not None and not os.path.exists(args.media_path):
            raise ValueError(f"Media file not found: {args.media_path}")
        # For t2v/t2i tasks, media_path is optional (pure generation)
        if args.task_type not in ["t2v", "t2i"] and args.media_path is None:
            raise ValueError(
                f"--media_path is required for task type '{args.task_type}'. "
                "Only t2v/t2i tasks can run without a media file."
            )
    elif csv_mode:
        if args.data_root_path is None:
            raise ValueError("--data_root_path is required when using --csv_path.")
    else:
        raise ValueError(
            "Please specify either single sample mode (--media_path + --prompt) "
            "or CSV batch mode (--csv_path + --data_root_path)."
        )

    # Set dtype based on argument
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    accelerator = Accelerator()

    # Load instruction rewrite model AFTER Accelerator so we know which GPU to use.
    # Must pass accelerator.device to avoid device_map="auto" spreading across all GPUs.
    rewrite_model = None
    rewrite_processor = None
    if args.rewrite_instruction:
        rewrite_model, rewrite_processor = load_rewrite_model(
            args.rewrite_model_path, device=accelerator.device
        )

    # Load models
    pipe = Capybara_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        transformer_version="capybara_v01",
        enable_offloading=True,
        enable_group_offloading=None,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        flow_shift=5.0,
        device=accelerator.device,
    )

    accelerator.wait_for_everyone()

    if single_sample_mode:
        run_single_sample(args, pipe, accelerator, rewrite_model, rewrite_processor)
    else:
        run_csv_batch(args, pipe, accelerator, rewrite_model, rewrite_processor)

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    print(f"\nAll processed successfully! Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()