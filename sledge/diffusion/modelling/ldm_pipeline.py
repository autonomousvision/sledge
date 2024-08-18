import inspect
from typing import List, Optional, Union

import torch

from diffusers.models import DiTTransformer2DModel
from diffusers.schedulers import DDPMScheduler
from diffusers import DiffusionPipeline

from sledge.autoencoder.modeling.models.rvae.rvae_decoder import RVAEDecoder
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector


class LDMPipeline(DiffusionPipeline):
    """Latent Diffusion Model pipeline for generation."""

    def __init__(self, decoder: RVAEDecoder, transformer: DiTTransformer2DModel, scheduler: DDPMScheduler):
        """
        Initializes diffusion pipeline.
        :param decoder: decoder module of raster-vector autoencoder
        :param transformer: diffusion transformer
        :param scheduler: noise schedular
        """
        super().__init__()
        self.register_modules(decoder=decoder, transformer=transformer, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        num_inference_timesteps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_classes: int = 4,
    ) -> List[SledgeVector]:
        """
        Generates a batch of sledge vectors.
        :param class_labels: list of integers for classes to generate.
        :param num_inference_timesteps: iterative diffusion steps, defaults to 50
        :param guidance_scale: scale for classifier-free guidance, defaults to 4.0
        :param generator: optional torch generator, defaults to None
        :param eta: noise multiplier, defaults to 0.0
        :param num_classes: number of classes, defaults to 4
        :return: list of sledge vector dataclass
        """

        batch_size = len(class_labels)
        class_labels = torch.tensor(class_labels, device=self.device).reshape(-1)
        class_null = torch.tensor([num_classes] * batch_size, device=self.device)
        class_labels_input = torch.cat([class_labels, class_null], 0)

        latents = torch.randn(
            (
                batch_size,
                self.transformer.config.in_channels,
                self.transformer.config.sample_size,
                self.transformer.config.sample_size,
            ),
            generator=generator,
            device=self.device,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = latent_model_input * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_timesteps, device=self.device)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):

            # scale the model input
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_prediction = self.transformer(
                hidden_states=latent_model_input,
                class_labels=class_labels_input,
                timestep=t.unsqueeze(0),
            ).sample

            # perform guidance
            cond_eps, uncond_eps = torch.split(noise_prediction, len(noise_prediction) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            noise_prediction = torch.cat([half_eps, half_eps], dim=0)

            # compute the previous noisy sample x_t -> x_t-1
            latent_model_input = self.scheduler.step(
                noise_prediction, t, latent_model_input, **extra_kwargs
            ).prev_sample

        # split the latent into the two halves
        latent_model_input, _ = latent_model_input.chunk(2, dim=0)

        # convert the vectors to images
        vector_output = self.decoder.decode(latent_model_input).unpack()

        return vector_output
