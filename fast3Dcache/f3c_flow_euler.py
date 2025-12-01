# fast3Dcache/f3c_flow_euler.py

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from trellis.modules.spatial import unpatchify, patchify
from .f3c_leader import LEADER
from .selection import AdvancedStabilityTracker

class F3cFlowEulerCfgSampler(FlowEulerGuidanceIntervalSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_dtype = torch.float16 
        self.stability_tracker = AdvancedStabilityTracker(num_tokens=LEADER.resolution ** 3, resolution=LEADER.resolution)

    def _init_f3c_state(self, x_t, args, model):
        LEADER.set_parameters(args)
        if hasattr(model, 'dtype'):
             self.model_dtype = model.dtype
        elif hasattr(model, 'parameters'):
            try:
                self.model_dtype = next(model.parameters()).dtype
            except StopIteration:
                pass
        
        self.stability_tracker.reset(device=x_t.device, latent_channels=x_t.shape[1])
        
    def _run_model_core(self, tokens: torch.Tensor, t_tensor: torch.Tensor, cond: torch.Tensor, model: nn.Module) -> torch.Tensor:
        target_dtype = self.model_dtype 
        original_dtype = tokens.dtype
        # 1. Calculate Timestep Embedding
        t_emb = model.t_embedder(t_tensor) 
        # 2. Apply AdaLN Modulation if it exists
        if hasattr(model, 'share_mod') and model.share_mod and hasattr(model, 'adaLN_modulation'):
            t_emb = model.adaLN_modulation(t_emb)
        # 3. Convert all inputs to target dtype for blocks
        t_emb = t_emb.to(target_dtype)
        h = tokens.to(target_dtype) 
        cond = cond.to(target_dtype)
        # 4. Run Transformer Blocks
        for block in model.blocks:
            h = block(h, t_emb, cond) 
        # 5. Convert back to original dtype (float32) before final layers
        h = h.to(original_dtype)
        # 6. Apply the final LayerNorm (on float32 tensor)
        h = F.layer_norm(h, h.shape[-1:])
        # 7. Apply the output layer (expects float32)
        pred_v_tokens = model.out_layer(h)
        return pred_v_tokens
    
    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps, cfg_strength, decoder, args, verbose=True, cfg_interval: Tuple[float, float] = (0.5, 1.0), **kwargs):
        self._init_f3c_state(noise, args, model) 
        B, C_in, D, H, W = noise.shape
        total_tokens = (D // model.patch_size) ** 3
        out_patched_channels = C_in 
        if hasattr(model, 'out_channels') and hasattr(model, 'patch_size') and model.patch_size > 0:
             out_patched_channels = model.out_channels * model.patch_size**3
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        rescale_t_val = 3.0
        if rescale_t_val != 1.0:
            t_seq = rescale_t_val * t_seq / (1 + (rescale_t_val - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_0_latents": []})
        last_pred_v_grid = None
        cached_v_tokens = torch.zeros(B, total_tokens, out_patched_channels, device=noise.device, dtype=torch.float32)
        for t, t_prev in tqdm(t_pairs, desc="F3C Sampler", disable=not verbose):
            current_step = LEADER.current_step
            
            is_f3c_active = False
            cached_indices, fast_update_indices = None, None
            if args.use_f3c and last_pred_v_grid is not None and current_step >= LEADER.full_sampling_steps:
                num_to_skip = LEADER.get_skip_budget_for_current_step(t)
                if num_to_skip > 0 and num_to_skip < total_tokens:
                    is_f3c_active = True
                    cached_indices, fast_update_indices = self.stability_tracker.update_and_select(last_pred_v_grid, num_to_skip, t)

            h = patchify(sample.float(), model.patch_size) 
            h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
            
            input_tokens_full = model.input_layer(h) + model.pos_emb[None] 
            input_tokens = input_tokens_full[:, fast_update_indices, :] if is_f3c_active else input_tokens_full
            
            t_tensor = torch.tensor([1000 * t] * sample.shape[0], device=sample.device)
            
            pred_v_tokens = None
            cond_ = cond if cond.shape[0] == B else cond.repeat(B, *([1]*(cond.ndim-1)))

            if cfg_interval[0] <= t <= cfg_interval[1]:
                neg_cond_ = neg_cond if neg_cond.shape[0] == B else neg_cond.repeat(B, *([1]*(neg_cond.ndim-1)))
                cond_pred_v_tokens = self._run_model_core(input_tokens, t_tensor, cond_, model)
                uncond_pred_v_tokens = self._run_model_core(input_tokens, t_tensor, neg_cond_, model)
                pred_v_tokens = uncond_pred_v_tokens + cfg_strength * (cond_pred_v_tokens - uncond_pred_v_tokens)
            else:
                pred_v_tokens = self._run_model_core(input_tokens, t_tensor, cond_, model)

            if is_f3c_active:
                final_v_tokens = cached_v_tokens.clone()
                final_v_tokens[:, fast_update_indices, :] = pred_v_tokens
            else:
                final_v_tokens = pred_v_tokens

            grid_size = D // model.patch_size 
            
            expected_unpatch_shape = (B, out_patched_channels, grid_size, grid_size, grid_size)
            reshaped_tokens = final_v_tokens.permute(0, 2, 1).view(*expected_unpatch_shape)
            
            current_v_grid = unpatchify(reshaped_tokens, model.patch_size).contiguous()
            
            sample = sample - (t - t_prev) * current_v_grid.to(sample.dtype) 
            
            latent_x0, _ = self._v_to_xstart_eps(x_t=sample.float(), t=t_prev, v=current_v_grid.float())
            ret.pred_x_0_latents.append(latent_x0)
            
            last_pred_v_grid = current_v_grid.float()
            cached_v_tokens = final_v_tokens
            
            if args.use_f3c and current_step == LEADER.anchor_step:
                if current_step > 0 and len(ret.pred_x_0_latents) > current_step:
                     prev_latent_x0 = ret.pred_x_0_latents[current_step - 1]
                     try:
                         decoder_device = next(decoder.parameters()).device
                         grid_t = (decoder(latent_x0.to(decoder_device)) > 0) 
                         grid_t_minus_1 = (decoder(prev_latent_x0.to(decoder_device)) > 0)
                         total_changes = torch.sum(grid_t != grid_t_minus_1).item()
                         LEADER.record_complexity_at_anchor(total_changes)
                     except Exception as e:
                         print(f"Error during decoder call at anchor step: {e}")

            LEADER.increase_step()
        
        ret.samples = sample
        return ret