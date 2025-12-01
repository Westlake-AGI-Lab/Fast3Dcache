# fast3Dcache/example_f3c.py
import sys
import os
os.environ['SPCONV_ALGO'] = 'native'

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from PIL import Image
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from fast3Dcache.f3c_argparser import parse_f3c_args
from fast3Dcache.f3c_trellis_pipeline import update_trellis_pipeline_for_f3c

def main():
    args = parse_f3c_args()
    
    print("Loading Trellis pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    print("Pipeline loading completed.")
    image = Image.open(args.image_path)

    if args.use_f3c:
        print("Fast3Dcache (F3C) Loading...")
        pipeline = update_trellis_pipeline_for_f3c(pipeline, args)
        
        sampler_params = {
              "steps": args.effective_steps,
              "cfg_strength": 7.5,
              "decoder": pipeline.models['sparse_structure_decoder'],
              "args": args
        }
        print("Use F3cFlowEulerCfgSampler")

    else:
        print("Fast3Dcache (F3C) is not used。")
        sampler_params = {
              "steps": args.effective_steps,
              "cfg_strength": 7.5
        }
        print("Use FlowEulerGuidanceIntervalSampler")


    print(f"Using seed {args.seed} and {args.effective_steps} steps to generate...")

    outputs = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params=sampler_params
    )
    
    if 'gaussian' not in outputs or 'mesh' not in outputs:
         return

    print("Post-processing is underway and results are being generated -> glb ...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, f"{args.output_name}.glb")

    glb.export(output_path)
    print(f"Finished. The result is saved in {output_path}")

if __name__ == "__main__":
    main()