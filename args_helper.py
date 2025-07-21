import argparse
import os

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data/vjuicefs_ai_camera_pgroup_ql/apps/portrait_light/common/models/black-forest-labs/FLUX.1-Fill-dev",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
        
    return args