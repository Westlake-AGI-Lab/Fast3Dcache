# fast3Dcache/f3c_flow_euler.py

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
import math
import traceback
from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from trellis.modules.spatial import unpatchify, patchify
from .f3c_leader import LEADER
from .selection import AdvancedStabilityTracker

def count_linear_flops(in_features, out_features, B, N, bias=True):
    flops = B * N * in_features * out_features * 2.0
    if bias: flops += B * N * out_features
    return flops
def count_layernorm_flops(B, N, C):
    return B * N * C * 7.0
def count_silu_flops(B, N, C):
    return B * N * C * 5.0
def count_attention_flops(B, N_q, N_kv, D_model, num_heads):
    flops = 0.0
    flops += count_linear_flops(D_model, D_model * 3, B, N_q, bias=True)
    dk = D_model // num_heads
    flops += B * num_heads * N_q * N_kv * dk * 2
    flops += B * num_heads * N_q * N_kv * 5 
    flops += B * num_heads * N_q * N_kv * dk * 2
    flops += count_linear_flops(D_model, D_model, B, N_q, bias=True)
    return flops
def count_mlp_flops(B, N, D_model, mlp_ratio, bias=True):
    D_mlp = int(D_model * mlp_ratio)
    flops = 0.0
    flops += count_linear_flops(D_model, D_mlp, B, N, bias=bias)
    flops += B * N * D_mlp * 5.0 
    flops += count_linear_flops(D_mlp, D_model, B, N, bias=bias) 
    return flops

class F3cFlowEulerCfgSampler(FlowEulerGuidanceIntervalSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_dtype = torch.float16
        self.stability_tracker = AdvancedStabilityTracker(num_tokens=LEADER.resolution ** 3, resolution=LEADER.resolution)
        self.last_run_flops = 0.0

    def _init_f3c_state(self, x_t, args, model):
        LEADER.set_parameters(args)
        current_num_tokens = LEADER.resolution ** 3
        if not hasattr(self.stability_tracker, 'num_tokens') or self.stability_tracker.num_tokens != current_num_tokens:
            self.stability_tracker = AdvancedStabilityTracker(num_tokens=current_num_tokens, resolution=LEADER.resolution)
        if hasattr(model, 'dtype'):
            self.model_dtype = model.dtype
        elif hasattr(model, 'parameters') and any(True for _ in model.parameters()):
            self.model_dtype = next(model.parameters()).dtype
        else:
            self.model_dtype = torch.float32
        self.stability_tracker.reset(device=x_t.device, latent_channels=x_t.shape[1])
        self.stability_tracker.set_hyperparameters(args)
        print(f"[F3C Setup] F3C state initialized. Model dtype: {self.model_dtype}. Tracker device: {self.stability_tracker.device}.")

    def _run_model_core(self, tokens: torch.Tensor, t_tensor: torch.Tensor, cond: torch.Tensor, model: nn.Module) -> Tuple[torch.Tensor, float]:
        """Runs the core transformer blocks and output layer ON ACTIVE TOKENS, returning prediction and estimated FLOPs."""
        
        core_flops = 0.0
        target_dtype = self.model_dtype
        original_dtype = tokens.dtype
        B, N_active, C_token_in = tokens.shape 
        D_model = getattr(model, 'model_channels', C_token_in) 
        D_cond = getattr(model, 'cond_channels', cond.shape[-1])
        num_heads = getattr(model, 'num_heads', 8)
        mlp_ratio = getattr(model, 'mlp_ratio', 4.0)
        N_cond = cond.shape[1]
        D_out_patch_actual = getattr(model.out_layer, 'out_features', 8) 

        with torch.no_grad():
            t_emb_tensor_only = model.t_embedder(t_tensor)
        mod_signal = t_emb_tensor_only
        if getattr(model, 'share_mod', False):
            mod_signal = model.adaLN_modulation(t_emb_tensor_only)

        mod_signal = mod_signal.to(target_dtype)
        h = tokens.to(target_dtype)
        cond = cond.to(target_dtype)

        for i, block in enumerate(model.blocks):
            block_flops_estimate = self.calculate_block_flops_static(model, B, N_active, N_cond)
            core_flops += block_flops_estimate
            h = block(h, mod_signal, cond)

        h = h.to(original_dtype)

        final_norm_flops = count_layernorm_flops(B, N_active, D_model)
        core_flops += final_norm_flops
        h = F.layer_norm(h.float(), (D_model,), eps=1e-6).to(original_dtype)

        output_flops = count_linear_flops(D_model, D_out_patch_actual, B, N_active, model.out_layer.bias is not None)
        core_flops += output_flops
        pred_v_tokens = model.out_layer(h)
        
        return pred_v_tokens, core_flops

    @staticmethod
    def calculate_block_flops_static(model, B, N_active, N_cond):
        """Static method to estimate FLOPs for one Transformer block based on model config."""
        flops = 0.0
        D_model = getattr(model, 'model_channels', 1024)
        num_heads = getattr(model, 'num_heads', 16)
        mlp_ratio = getattr(model, 'mlp_ratio', 4.0)
        has_bias = True

        if not getattr(model, 'share_mod', True):
             flops += count_silu_flops(B, 1, D_model)
             flops += count_linear_flops(D_model, 6 * D_model, B, 1, has_bias)

        flops += count_layernorm_flops(B, N_active, D_model)
        flops += B * N_active * D_model * 2.0
        flops += count_attention_flops(B, N_active, N_active, D_model, num_heads)
        flops += B * N_active * D_model
        flops += B * N_active * D_model
        flops += count_layernorm_flops(B, N_active, D_model)
        flops += count_attention_flops(B, N_active, N_cond, D_model, num_heads)
        flops += B * N_active * D_model
        flops += count_layernorm_flops(B, N_active, D_model)
        flops += B * N_active * D_model * 2.0
        flops += count_mlp_flops(B, N_active, D_model, mlp_ratio, has_bias)
        flops += B * N_active * D_model
        flops += B * N_active * D_model
        
        return flops

    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps, cfg_strength, decoder, args, verbose=True, cfg_interval: Tuple[float, float] = (0.5, 1.0), **kwargs):
        self._init_f3c_state(noise, args, model) 
        total_core_flops = 0.0
        B, C_in, D_res, H_res, W_res = noise.shape
        total_tokens = D_res * H_res * W_res
        D_out_actual = getattr(model.out_layer, 'out_features', C_in) 
        
        total_flops = 0.0
        print(f"\n--- [F3C FLOPs] Starting Sampling Estimation for {steps} steps ---")
        cached_v_tokens = torch.zeros(B, total_tokens, D_out_actual, device=noise.device, dtype=torch.float32)

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        rescale_t_val = 3.0
        if rescale_t_val != 1.0:
            t_seq = rescale_t_val * t_seq / (1 + (rescale_t_val - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_0_latents": []})
        last_pred_v_grid = None

        for t, t_prev in tqdm(t_pairs, desc="F3C Sampler", disable=not verbose):
            step_flops = 0.0
            current_step = LEADER.current_step

            flops_input_layer = count_linear_flops(C_in, model.model_channels, B, total_tokens, model.input_layer.bias is not None)
            step_flops += flops_input_layer
            flops_posemb = B * total_tokens * model.model_channels
            step_flops += flops_posemb
            
            t_tensor = torch.tensor([1000 * t] * B, device=sample.device)
            d_t = model.t_embedder.mlp[0].in_features 
            flops_t_emb = count_linear_flops(d_t, d_t, B, 1, True) + B * 1 * d_t * 5 + count_linear_flops(d_t, model.model_channels, B, 1, True)
            step_flops += flops_t_emb
            if model.share_mod:
                 flops_mod = B * 1 * model.model_channels * 5 + count_linear_flops(model.model_channels, 6*model.model_channels, B, 1, True)
                 step_flops += flops_mod
        
            is_f3c_active = False
            N_active = total_tokens
            fast_update_indices = torch.arange(total_tokens, device=sample.device)

            if args.use_f3c and last_pred_v_grid is not None and current_step >= LEADER.full_sampling_steps:
                num_to_skip = LEADER.get_skip_budget_for_current_step(t)
                if num_to_skip > 0 and num_to_skip < total_tokens:
                    is_f3c_active = True
                    cached_indices, fast_update_indices = self.stability_tracker.update_and_select(last_pred_v_grid, num_to_skip, t)
                    N_active = fast_update_indices.numel()

            h = sample.float().view(B, C_in, -1).permute(0, 2, 1).contiguous()
            input_tokens_full = model.input_layer(h) + model.pos_emb[None]
            input_tokens = input_tokens_full[:, fast_update_indices, :]

            core_flops_this_step = 0.0
            cond_ = cond.repeat(B, *([1]*(cond.ndim-1))) if cond.shape[0] != B else cond
            neg_cond_ = neg_cond.repeat(B, *([1]*(neg_cond.ndim-1))) if neg_cond.shape[0] != B else neg_cond
            
            if cfg_interval[0] <= t <= cfg_interval[1]:
                uncond_pred_v_tokens, flops_uncond = self._run_model_core(input_tokens, t_tensor, neg_cond_, model)
                cond_pred_v_tokens, flops_cond = self._run_model_core(input_tokens, t_tensor, cond_, model)
                core_flops_this_step += 2 * flops_cond
                pred_v_tokens = uncond_pred_v_tokens + cfg_strength * (cond_pred_v_tokens - uncond_pred_v_tokens)
            else:
                pred_v_tokens, flops_single = self._run_model_core(input_tokens, t_tensor, cond_, model)
                core_flops_this_step += flops_single
            
            step_flops += core_flops_this_step
            total_core_flops += core_flops_this_step
            print(f" [Step {current_step+1}] Fixed FLOPs: {step_flops - core_flops_this_step:.3e}, Core FLOPs (N_active={N_active}): {core_flops_this_step:.3e}, Total Step FLOPs: {step_flops:.3e}")

            if is_f3c_active:
                if cached_v_tokens.shape[-1] != pred_v_tokens.shape[-1]:
                     D_out_actual = pred_v_tokens.shape[-1]
                     cached_v_tokens = torch.zeros(B, total_tokens, D_out_actual, device=noise.device, dtype=torch.float32)
                
                final_v_tokens = cached_v_tokens.clone()
                final_v_tokens[:, fast_update_indices, :] = pred_v_tokens.to(cached_v_tokens.dtype)
            else:
                final_v_tokens = pred_v_tokens.to(cached_v_tokens.dtype)

            D_out_actual = final_v_tokens.shape[-1]
            current_v_grid = final_v_tokens.permute(0, 2, 1).view(B, D_out_actual, D_res, H_res, W_res).contiguous()
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
                        print(f" [Step {current_step+1} ANCHOR] Recorded complexity: {total_changes}")
                    except Exception as e:
                        print(f"Error during decoder call at anchor step: {e}")

            total_flops += step_flops
            
            LEADER.increase_step()

        ret.samples = sample
        self.last_run_flops = total_core_flops 
        
        ret.total_core_flops = total_core_flops 
        print(f"\n--- [F3C FLOPs] Sampling Complete ---")
        print(f"[F3C FLOPs] Final Estimated Total CORE FLOPs ({steps} steps): {total_core_flops:.4e} ---") 
        return ret