import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusionXLPipeline, UNet2DConditionModel, StableDiffusionUpscalePipeline
from diffusers.models.transformers import SD3Transformer2DModel
from .models.qcache_sd3_transformer import QCacheSD3Transformer
from .utils import QCacheConfig, PatchCacheManager
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from typing import Optional, Union
from diffusers.utils import is_accelerate_available, is_accelerate_version


class QCacheSD3Pipeline:
    def __init__(self, pipeline: StableDiffusion3Pipeline, module_config: QCacheConfig):
        self.pipeline = pipeline
        self.qcache_config = module_config

        self.static_inputs = None

        #self.prepare()

    @staticmethod
    def from_pretrained(qcache_config: QCacheConfig, **kwargs):
        device = qcache_config.device
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path"
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        transformer = SD3Transformer2DModel.from_pretrained(
            'stabilityai/stable-diffusion-3-medium-diffusers', torch_dtype=torch_dtype, subfolder="transformer"
        ).to(device)

        transformer = QCacheSD3Transformer(transformer, qcache_config)
        
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, transformer = transformer, **kwargs
        ).to(device)
        return QCacheSD3Pipeline(pipeline, qcache_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert "height" not in kwargs, "height should not be in kwargs"
        assert "width" not in kwargs, "width should not be in kwargs"
        config = self.qcache_config
        if not config.do_classifier_free_guidance:
            if "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
        self.pipeline.transformer.set_counter(0)
        self.pipeline.transformer.clear_model()
        return self.pipeline(height=config.height, width=config.width, *args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        
        #assert "height" not in kwargs, "height should not be in kwargs"
        #assert "width" not in kwargs, "width should not be in kwargs"
        height = kwargs.pop("height", self.pipeline.default_sample_size * self.pipeline.vae_scale_factor)
        width = kwargs.pop("width", self.pipeline.default_sample_size * self.pipeline.vae_scale_factor)
        
        config = self.qcache_config
        setattr(config, 'prompt', kwargs.get('prompt', None))
        return_dict = kwargs.pop("return_dict", False)  
        if not config.do_classifier_free_guidance:
            if "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
                
        pc_manager = PatchCacheManager(config)
        pc_manager.sigmas = self.pipeline.scheduler.sigmas
        
        self.pipeline.transformer.set_cache_manager(pc_manager)
        
        self.pipeline.transformer.set_counter(0)
        self.pipeline.transformer.clear_model()
      
        #height = self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        #width = self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        prompt = kwargs.pop("prompt", None)
        prompt_2 = kwargs.pop("prompt_2", None)
        prompt_3 = kwargs.pop("prompt_3", None)
        num_inference_steps = kwargs.pop("num_inference_steps", 20)
        timesteps = kwargs.pop("time_steps", None)
        guidance_scale = kwargs.pop("guidance_scale", 7.0)
        negative_prompt = kwargs.pop("negative_prompt", None)
        negative_prompt_2 = kwargs.pop("negative_prompt_2", None)
        negative_prompt_3 = kwargs.pop("negative_prompt_3", None)
        num_images_per_prompt = kwargs.pop("num_images_per_prompt", 1)
        generator = kwargs.pop("generator", None)
        latents = kwargs.pop("latents", None)
        prompt_embeds = kwargs.pop("prompt_embeds", None)
        negative_prompt_embeds = kwargs.pop("negative_prompt_embeds", None)
        pooled_prompt_embeds = kwargs.pop("pooled_prompt_embeds", None)
        negative_pooled_prompt_embeds = kwargs.pop("negative_pooled_prompt_embeds", None)
        output_type = kwargs.pop("output_type", "pil")
    
        joint_attention_kwargs = kwargs.pop("joint_attention_kwargs", None)
        clip_skip = kwargs.pop("clip_skip", None)
        callback_on_step_end = kwargs.pop("callback_on_step_end", None)
        callback_on_step_end_tensor_inputs = kwargs.pop("callback_on_step_end_tensor_inputs", ["latents"])
        max_sequence_length = kwargs.pop("max_sequence_length", 256)
        
        

    #    print(f'height: {height}')
        # 1. Check inputs. Raise error if not correct
        self.pipeline.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._clip_skip = clip_skip
        self.pipeline._joint_attention_kwargs = joint_attention_kwargs
        self.pipeline._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipeline._execution_device

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.pipeline.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.pipeline.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if self.pipeline.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.pipeline.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pipeline.scheduler.order, 0)
        self.pipeline._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.pipeline.transformer.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        #print(self.pipeline.scheduler)
        import time
        torch.cuda.synchronize()
        start = time.time()

        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                if self.pipeline.interrupt:
                    continue
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.pipeline.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

             
                noise_pred = self.pipeline.transformer.forward(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                    return_dict=False,
                    record = False,
                  
                )[0]

        
              
                # perform guidance
                if self.pipeline.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
               # print(f'noise shape: {noise_pred.shape}')
                latents = self.pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
      
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                    progress_bar.update()

        torch.cuda.synchronize()
        print(f'time taken: {time.time() - start}') 

        self.pipeline.transformer.clear_model()
        torch.cuda.empty_cache()
        images = []
        
        num_images = latents.shape[0]

        for i in range(num_images):
            latent = latents[i].unsqueeze(0)
            latent = (latent / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
            image = self.pipeline.vae.decode(latent, return_dict=False)[0]
            image = self.pipeline.image_processor.postprocess(image, output_type='pil')
            images.append(image[0])
            
        self.pipeline.maybe_free_model_hooks()
        
        if not return_dict:
            return tuple(images)
        return StableDiffusion3PipelineOutput(images=images)
