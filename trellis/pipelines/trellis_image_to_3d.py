# trellis/pipelines/trellis_image_to_3d.py

from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp

class TrellisImageTo3DPipeline(Pipeline):
    def __init__(self, models: dict[str, nn.Module] = None, sparse_structure_sampler: samplers.Sampler = None, slat_sampler: samplers.Sampler = None, slat_normalization: dict = None, image_cond_model: str = None):
        if models is None: return
        super().__init__(models)
        self.sparse_structure_sampler, self.slat_sampler = sparse_structure_sampler, slat_sampler
        self.sparse_structure_sampler_params, self.slat_sampler_params = {}, {}
        self.slat_normalization, self.rembg_session = slat_normalization, None
        self._init_image_cond_model(image_cond_model)
    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']
        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']
        new_pipeline.slat_normalization = args['slat_normalization']
        new_pipeline._init_image_cond_model(args['image_cond_model'])
        return new_pipeline
    def _init_image_cond_model(self, name: str):
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        self.image_cond_model_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    def preprocess_image(self, input: Image.Image) -> Image.Image:
        has_alpha = input.mode == 'RGBA' and not np.all(np.array(input)[:, :, 3] == 255)
        if not has_alpha:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1: input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None: self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        else: output = input
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox).resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        return Image.fromarray((output * 255).astype(np.uint8))
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        if isinstance(image, list):
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [torch.from_numpy(np.array(i.convert('RGB')).astype(np.float32) / 255).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image)
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        return F.layer_norm(features, features.shape[-1:])
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        cond = self.encode_image(image)
        return {'cond': cond, 'neg_cond': torch.zeros_like(cond)}

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[list]]]:
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        sampler = self.sparse_structure_sampler

        z_s_result = sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )
        z_s = z_s_result.samples

        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        
        if 'occupancy_grids' in z_s_result or 'noise_predictions' in z_s_result:
             return coords, z_s_result
        else:
             return coords, None

    def decode_slat(self, slat: sp.SparseTensor, formats: List[str] = ['mesh', 'gaussian', 'radiance_field']) -> dict:
        ret = {}
        if 'mesh' in formats: ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats: ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats: ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    def sample_slat(self, cond: dict, coords: torch.Tensor, sampler_params: dict = {}) -> sp.SparseTensor:
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device), coords=coords)
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=True).samples
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        return slat * std + mean
        
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True
    ) -> dict:
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        
        coords, z_s_result = self.sample_sparse_structure(
            cond, num_samples, sparse_structure_sampler_params
        )
        
        if z_s_result and ('occupancy_grids' in z_s_result or 'noise_predictions' in z_s_result):
            return z_s_result

        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
        
    @contextmanager
    def inject_sampler_multi_image(self, sampler_name: str, num_images: int, num_steps: int, mode: Literal['stochastic', 'multidiffusion'] = 'stochastic'):
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)
        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. This may lead to performance degradation.\033[0m")
            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = [FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs) for i in range(len(cond))]
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = [FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs) for i in range(len(cond))]
                    return sum(preds) / len(preds)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))
        yield
        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')
    @torch.no_grad()
    def run_multi_image(self, images: List[Image.Image], num_samples: int = 1, seed: int = 42, sparse_structure_sampler_params: dict = {}, slat_sampler_params: dict = {}, formats: List[str] = ['mesh', 'gaussian', 'radiance_field'], preprocess_image: bool = True, mode: Literal['stochastic', 'multidiffusion'] = 'stochastic') -> dict:
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)