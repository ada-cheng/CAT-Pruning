import torch

from qcache.pipelines import QCacheSD3Pipeline
from qcache.utils import QCacheConfig

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

prompts = ['A girl, shirt, white hair, red eyes, earrings, Detailed face']*3 

stop_idx_dict = {   
    'joint_attn': 5,
    'ff': 5,
    'proj': 5,
}

select_mode = {
    'joint_attn':'convergence_t_noise',
    'ff' : 'convergence_t_noise',
    'proj': 'convergence_stale_cpp' # cpp is abbreviation for cluster + pooling * 2
}

select_factor = {   
    'joint_attn': 0.3,
    'ff': 0.3,
    'proj': 0.3
}

qcache_config = QCacheConfig(height=1024, width=1024, 
                             select_mode= select_mode,
                             stop_idx_dict=stop_idx_dict,
                             select_factor=select_factor)

pipeline = QCacheSD3Pipeline.from_pretrained(
    qcache_config=qcache_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers",
    variant="fp16",
    use_safetensors=True,
)


pipeline.set_progress_bar_config(disable=False)


torch.manual_seed(3407)
torch.cuda.manual_seed(3407)


for i, prompt in enumerate(prompts):
    image = pipeline.generate(
        prompt=prompt,
        generator=torch.Generator(device="cuda").manual_seed(3407),
        num_inference_steps=28,
        guidance_scale=7.0,
    )[0]
    image.save(f'girl.png')