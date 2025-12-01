# fast3Dcache/f3c_trellis_pipeline.py
def update_trellis_pipeline_for_f3c(pipeline, args):
    original_sampler = pipeline.sparse_structure_sampler
    from .f3c_flow_euler import F3cFlowEulerCfgSampler
    f3c_sampler = F3cFlowEulerCfgSampler(sigma_min=original_sampler.sigma_min)
    mode_str = "Euler + Fast3Dcache"
    pipeline.sparse_structure_sampler = f3c_sampler
    return pipeline