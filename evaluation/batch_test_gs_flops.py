# File: batch_test_gs_flops.py (Recursive Image Search)

import sys
import os
import glob
import argparse
import traceback
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

os.environ['SPCONV_ALGO'] = 'native'

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root) 

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis_flops.f3c_argparser import parse_f3c_args 
from trellis_flops.f3c_flow_euler import F3cFlowEulerCfgSampler 
from trellis_flops.f3c_trellis_pipeline import update_trellis_pipeline_for_f3c

def parse_batch_args():
    """ Parses arguments, modifying defaults for batch processing. """
    parser = argparse.ArgumentParser(
        description="Batch Test GS Stage FLOPs for Trellis with Optional F3C",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Root directory containing object folders with images (e.g., /path/to/TOY4K_transparent1).")

    f3c_strategy_group = parser.add_argument_group('Fast3Dcache Scheduling Strategy')
    f3c_strategy_group.add_argument("--use_f3c", action="store_true", help="Enable Fast3Dcache (F3C) acceleration.")
    f3c_strategy_group.add_argument("--euler_steps", type=int, default=25, help="Total steps for the Euler sampler (GS stage).")
    f3c_strategy_group.add_argument("--anchor_ratio", type=float, default=None, help="F3C anchor step ratio.")
    f3c_strategy_group.add_argument("--assumed_slope", type=float, default=None, help="F3C predicted slope.")
    f3c_strategy_group.add_argument("--full_sampling_ratio", type=float, default=0.2, help="F3C initial full sampling ratio.")
    f3c_strategy_group.add_argument("--full_sampling_end_ratio", type=float, default=0.75, help="F3C start ratio for final phase.")
    f3c_strategy_group.add_argument("--aggressive_cache_ratio", type=float, default=0.7, help="F3C aggressive cache ratio in final phase.")
    f3c_strategy_group.add_argument("--final_phase_correction_freq", type=int, default = 3, help="F3C correction frequency in final phase.")
    
    trellis_group = parser.add_argument_group('Trellis Internal Options')
    trellis_group.add_argument("--seed", type=int, default=42, help="Global random seed (for reproducibility if needed).")
    trellis_group.add_argument("--resolution", type=int, default=16, help="Internal resolution for GS stage model.") 
    
    args = parser.parse_args()
    args.effective_steps = args.euler_steps 
    if args.assumed_slope is None: args.assumed_slope = -0.07
    if args.anchor_ratio is None: args.anchor_ratio = 0.2
    if args.use_f3c:
        import math
        args.full_sampling_steps = math.floor(args.effective_steps * args.full_sampling_ratio)
        args.full_sampling_end_steps = math.ceil(args.effective_steps * args.full_sampling_end_ratio)
        calculated_anchor_step = math.floor(args.effective_steps * args.anchor_ratio)
        args.anchor_step = max(1, calculated_anchor_step)
        
    return args

def main():
    args = parse_batch_args()
    
    print("--- Loading Trellis Pipeline ---")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large") 
    pipeline.cuda()
    print("Pipeline loaded.")

    sampler_params = {
        "steps": args.effective_steps,
        "cfg_strength": 7.5 
    }
    
    if args.use_f3c:
        print("F3C Enabled. Replacing sampler...")
        original_sampler = pipeline.sparse_structure_sampler
        f3c_sampler = F3cFlowEulerCfgSampler(sigma_min=original_sampler.sigma_min) 
        pipeline.sparse_structure_sampler = f3c_sampler
        print(f"Sampler replaced with {type(f3c_sampler).__name__}.")
        sampler_params["decoder"] = pipeline.models['sparse_structure_decoder']
        sampler_params["args"] = args 
    else:
        print("F3C Disabled. Using default sampler.")
        from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
        if not isinstance(pipeline.sparse_structure_sampler, FlowEulerGuidanceIntervalSampler):
             print(f"Warning: Default sampler is not FlowEulerGuidanceIntervalSampler, but {type(pipeline.sparse_structure_sampler)}. FLOPs retrieval might fail.")

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp') 
    image_files = []
    print(f"Searching for images recursively in: {args.image_dir}")
    for ext in image_extensions:
        pattern = os.path.join(args.image_dir, '**', ext) 
        image_files.extend(glob.glob(pattern, recursive=True))
        
    if not image_files:
        print(f"Error: No images found recursively in directory: {args.image_dir}")
        return
        
    print(f"Found {len(image_files)} images.")

    all_core_flops = []
    failed_images = []
    
    torch.manual_seed(args.seed) 
    
    for img_path in tqdm(image_files, desc="Processing Images (GS Stage)"):
        try:
            image = Image.open(img_path)
            if image.mode != 'RGBA':
                 image = image.convert('RGBA')
            
            processed_image = pipeline.preprocess_image(image)
            cond = pipeline.get_cond([processed_image])
            
            _ = pipeline.sample_sparse_structure(
                cond, 
                num_samples=1, 
                sampler_params=sampler_params 
            ) 
            
            current_sampler = pipeline.sparse_structure_sampler
            if hasattr(current_sampler, 'last_run_flops'):
                flops_for_image = current_sampler.last_run_flops
                current_sampler.last_run_flops = 0.0 
                all_core_flops.append(flops_for_image)
            else:
                print(f"Warning: Could not retrieve FLOPs for {os.path.basename(img_path)}. Sampler type: {type(current_sampler)} lacks 'last_run_flops'.")
                failed_images.append(os.path.basename(img_path))
                
        except Exception as e:
            print(f"\nError processing {os.path.basename(img_path)}: {e}")
            traceback.print_exc() 
            failed_images.append(os.path.basename(img_path))
            
    print("\n--- Batch GS Stage FLOPs Test Results ---")
    num_processed = len(all_core_flops)
    num_total = len(image_files)
    num_failed = len(failed_images)

    if num_processed > 0:
        average_flops = np.mean(all_core_flops)
        print(f"Successfully processed {num_processed} / {num_total} images.")
        if num_failed > 0:
             print(f"Failed to process {num_failed} images: {failed_images}")
        print(f"Average CORE FLOPs per image (GS Stage Only): {average_flops:.4e}") 
    else:
        print("No FLOPs data was collected.")
        if num_failed > 0:
             print(f"Failed to process {num_failed} images: {failed_images}")
        print("Please ensure the correct sampler is used and FLOPs calculation ran.")

if __name__ == "__main__":
    main()