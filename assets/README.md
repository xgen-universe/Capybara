# Assets Directory

This directory contains example media files and test data for the Capybara inference pipeline.

## Directory Structure

```
assets/
├── examples/                   # Example media files for testing
│   ├── img1.jpeg               # Sample input image
│   ├── img2.jpeg               # Sample input image
│   ├── video1.mp4              # Sample input video
│   └── video2.mp4              # Sample input video
├── misc/                       # Miscellaneous assets
│   └── capybara.png            # Project logo
├── test_data/                  # Example CSV files for batch inference
│   ├── ti2i_example.csv        # CSV for instruction-based image editing (TI2I)
│   └── tv2v_example.csv        # CSV for instruction-based video editing (TV2V)
└── README.md
```

## CSV Format

CSV files are used for **batch inference mode**. Different task types use different column schemas:

### TI2I (Image Editing)

| Column        | Description                                          |
| ------------- | ---------------------------------------------------- |
| `img_path`    | Relative path to the input image (under data root)   |
| `instruction` | Text instruction describing the desired edit         |

Example (`ti2i_example.csv`):

```csv
img_path,instruction
img1.jpeg,Change the background to a beach.
img2.jpeg,Change the background to a beach.
```

### TV2V (Video Editing)

| Column        | Description                                          |
| ------------- | ---------------------------------------------------- |
| `video_path`  | Relative path to the input video (under data root)   |
| `instruction` | Text instruction describing the desired edit         |

Example (`tv2v_example.csv`):

```csv
video_path,instruction
video1.mp4,Add a fork
video2.mp4,Add a fork
```

## Usage

Use these example files with the batch inference script:

```bash
# Image editing (TI2I)
python inference.py \
    --pretrained_model_name_or_path ./ckpts \
    --csv_path ./assets/test_data/ti2i_example.csv \
    --data_root_path ./assets/examples \
    --output_path ./results/ti2i \
    --task_type ti2i \
    --resolution 480p \
    --num_frames 81

# Video editing (TV2V)
python inference.py \
    --pretrained_model_name_or_path ./ckpts \
    --csv_path ./assets/test_data/tv2v_example.csv \
    --data_root_path ./assets/examples \
    --output_path ./results/tv2v \
    --task_type tv2v \
    --resolution 480p \
    --num_frames 81
```

See `script/batch_inference.sh` and `script/single_inference.sh` for more complete examples.

## Notes

- Keep files small (< 10MB each) for GitHub compatibility.
- Large files should be hosted externally (e.g., Hugging Face) and downloaded separately.
- The `data_root_path` argument specifies the base directory for resolving relative paths in CSV files.
