# evaluation/calculate_throughput.py

import time
import torch
from PIL import Image
from tqdm import tqdm
import sys
import argparse
import numpy as np 

try:
    from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
    from fast3Dcache.f3c_trellis_pipeline import update_trellis_pipeline_for_f3c
except ImportError as e:
    sys.exit(1)

def calculate_throughput_for_image(pipeline: TrellisImageTo3DPipeline, image_path: str, args: argparse.Namespace) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        input_image = Image.open(image_path)
    except FileNotFoundError:
        return -1.0

    with torch.no_grad():
        image_preprocessed = pipeline.preprocess_image(input_image)
        cond_tensor = pipeline.get_cond([image_preprocessed])

    gs_sampler_params = {
        "steps": args.effective_steps,
        "cfg_strength": 7.5,
    }
    if args.use_f3c:
         gs_sampler_params["decoder"] = pipeline.models['sparse_structure_decoder']
         gs_sampler_params["args"] = args

    warmup_iters = 5
    for _ in range(warmup_iters):
        with torch.no_grad():
             _ = pipeline.sample_sparse_structure(cond=cond_tensor, num_samples=1, sampler_params=gs_sampler_params)

    if device == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(args.profile_iters):
        with torch.no_grad():
             _ = pipeline.sample_sparse_structure(cond=cond_tensor, num_samples=1, sampler_params=gs_sampler_params)
    
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency = total_time / args.profile_iters
    
    return avg_latency