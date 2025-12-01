# fast3Dcache/f3c_argparser.py 

import argparse
import math

def parse_f3c_args():

    parser = argparse.ArgumentParser(
        description="Trellis with Pluggable Fast3Dcache and Sampler Acceleration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    mode_group = parser.add_argument_group('Mode Control & Steps')
    mode_group.add_argument("--use_f3c", action="store_true", help="Fast3Dcache acceleration。")
    mode_group.add_argument("--euler_steps", type=int, default=25, help="--")
    
    f3c_strategy_group = parser.add_argument_group('Fast3Dcache Scheduling Strategy')
    f3c_strategy_group.add_argument("--anchor_ratio", type=float, default=None, 
                                     help="--")
    f3c_strategy_group.add_argument("--assumed_slope", type=float, default=None, help="--")
    f3c_strategy_group.add_argument("--full_sampling_ratio", type=float, default=0.2, 
                                     help="Phase 1 of Fast3Dcache。")
    f3c_strategy_group.add_argument("--full_sampling_end_ratio", type=float, default=0.75, 
                                     help="Phase 2 of Fast3Dcache")
    f3c_strategy_group.add_argument("--aggressive_cache_ratio", type=float, default=0.7, 
                                     help="Phase 3 of Fast3Dcache。")
    f3c_strategy_group.add_argument("--final_phase_correction_freq", type=int, default=3,
                                     help="f_corr")

    io_group = parser.add_argument_group('Trellis I/O & Internal Options')
    io_group.add_argument("--output_dir", type=str, default="outputs", help="--")
    io_group.add_argument("--image_path", type=str, default="/root/Fast3Dcache/assets/example_image/typical_creature_elephant.png", help="--。")
    io_group.add_argument("--output_name", type=str, default="sample_output", help="--")
    io_group.add_argument("--seed", type=int, default=42, help="SEED。")
    io_group.add_argument("--resolution", type=int, default=16, help="Gs Stage。")
    args = parser.parse_args()

    active_sampler = "Euler"; args.effective_steps = args.euler_steps
    if args.assumed_slope is None: args.assumed_slope = -0.07
    if args.anchor_ratio is None: args.anchor_ratio = 0.2
    if args.use_f3c:
        args.full_sampling_steps = math.floor(args.effective_steps * args.full_sampling_ratio)
        args.full_sampling_end_steps = math.ceil(args.effective_steps * args.full_sampling_end_ratio)
        calculated_anchor_step = math.floor(args.effective_steps * args.anchor_ratio)
        args.anchor_step = max(1, calculated_anchor_step)
    return args