# evaluation/eval.py

import sys
import os
import math
import argparse
from tqdm import tqdm
import numpy as np

os.environ['SPCONV_ALGO'] = 'native'

try:
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path)) 
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if project_root not in sys.path:
         sys.path.insert(0, project_root)
os.environ['SPCONV_ALGO'] = 'native'
from evaluation.calculate_throughput import calculate_throughput_for_image
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from fast3Dcache.f3c_trellis_pipeline import update_trellis_pipeline_for_f3c
def main():
    parser = argparse.ArgumentParser(
        description="--",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", type=str, default="microsoft/TRELLIS-image-large", help="--")
    parser.add_argument("--input_dir", type=str, required=True, help="--")
    parser.add_argument("--profile_iters", type=int, default=25, help="--")
    parser.add_argument("--seed", type=int, default=42, help="--")
    parser.add_argument("--resolution", type=int, default=16, help="Gs stage")
    mode_group = parser.add_argument_group('Mode Control & Steps')
    mode_group.add_argument("--use_f3c", action="store_true", help="--")
    mode_group.add_argument("--euler_steps", type=int, default=25, help="--")

    f3c_group = parser.add_argument_group('Fast3Dcache Scheduling Strategy')
    f3c_group.add_argument("--use_fixed_ras_ratio", action="store_true", help="--")
    f3c_group.add_argument("--fixed_ras_ratio", type=float, default=0.25, help="--")
    f3c_group.add_argument("--anchor_ratio", type=float, default=None, help="--")
    f3c_group.add_argument("--assumed_slope", type=float, default=None, help="--")
    f3c_group.add_argument("--full_sampling_ratio", type=float, default=0.2, help="Phase 1")
    f3c_group.add_argument("--full_sampling_end_ratio", type=float, default=0.75, help="Phase 2")
    f3c_group.add_argument("--aggressive_cache_ratio", type=float, default=0.7, help="Phase 3")
    f3c_group.add_argument("--final_phase_correction_freq", type=int, default=3, help="f_corr")
    
    args = parser.parse_args()
    args.effective_steps = args.euler_steps
    if args.assumed_slope is None: args.assumed_slope = -0.07
    if args.anchor_ratio is None: args.anchor_ratio = 0.2
    if args.use_f3c:
        args.full_sampling_steps = math.floor(args.effective_steps * args.full_sampling_ratio)
        args.full_sampling_end_steps = math.ceil(args.effective_steps * args.full_sampling_end_ratio)
        args.anchor_step = max(1, math.floor(args.effective_steps * args.anchor_ratio))
    
    print("\n" + "="*50)
    print("=== Begin ! ===")
    print("="*50)
    image_paths_to_test = []
    for root, dirs, files in os.walk(args.input_dir):
        if "0001.png" in files:
            parent_dir_name = os.path.basename(root)
            if "_" in parent_dir_name:
                    image_paths_to_test.append(os.path.join(root, "0001.png"))
    if not image_paths_to_test:
        sys.exit(1)

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()
    if args.use_f3c:
        pipeline = update_trellis_pipeline_for_f3c(pipeline, args)

    all_latencies = []
    for image_path in tqdm(image_paths_to_test, desc="Overall Progress"):
        avg_latency = calculate_throughput_for_image(pipeline, image_path, args) 
        if avg_latency > 0:
            all_latencies.append(avg_latency)
            category_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
            tqdm.write(f"  - {category_name}: {avg_latency:.4f} s/iter")
    if all_latencies:
        overall_avg_latency = np.mean(all_latencies)
        overall_avg_throughput = 1.0 / overall_avg_latency

        print("\n" + "="*50)
        print("--- test results ---")
        print(f"cases: {len(all_latencies)}")
        print(f"Latency: {overall_avg_latency:.4f} s/iter")
        print(f"Throughput: {overall_avg_throughput:.4f} iter/s")
        print("="*50)
        
    else:
        print("Fail.")

if __name__ == "__main__":
    main()