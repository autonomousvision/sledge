import math
import os
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F

from packaging import version
from tqdm.auto import tqdm

import accelerate
from accelerate.logging import get_logger

import datasets
import diffusers
from diffusers import DiTTransformer2DModel
from diffusers.training_utils import EMAModel
import pyarrow_hotfix

from sledge.diffusion.modelling.ldm_pipeline import LDMPipeline
from sledge.autoencoder.callbacks.rvae_visualization_callback import get_sledge_vector_as_raster
from sledge.script.builders.diffusion_builder import (
    build_accelerator,
    build_diffusion_model,
    build_noise_scheduler,
    build_optimizer,
    build_dataset,
    build_dataloader,
    build_lr_scheduler,
    build_decoder,
    build_pipeline_from_checkpoint,
)

pyarrow_hotfix.uninstall()
logger = get_logger(__name__, log_level="INFO")


def run_training_diffusion(cfg):

    # Build accelerator
    accelerator = build_accelerator(cfg)

    if cfg.diffusion_checkpoint is not None:
        # Build complete ldm pipeline
        pipeline = build_pipeline_from_checkpoint(cfg)
        diffusion_model, noise_scheduler, decoder = pipeline.transformer, pipeline.scheduler, pipeline.decoder

    else:
        # Build diffusion model
        diffusion_model = build_diffusion_model(cfg)

        # Build Noise Scheduler
        noise_scheduler = build_noise_scheduler(cfg)

        # build decoder
        decoder = build_decoder(cfg)

    # build optimizer
    optimizer = build_optimizer(cfg, diffusion_model)

    # build dataset
    dataset = build_dataset(cfg)

    # build dataloader
    dataloader = build_dataloader(cfg, dataset)

    # build lr_scheduler
    lr_scheduler = build_lr_scheduler(cfg, optimizer, len(dataloader) * cfg.num_epochs)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if cfg.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "transformer_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "transformer"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if cfg.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), DiTTransformer2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = DiTTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    # Create EMA for the model.
    if cfg.ema.use_ema:
        ema_model = EMAModel(
            diffusion_model.parameters(),
            decay=cfg.ema.max_decay,
            use_ema_warmup=True,
            inv_gamma=cfg.ema.inv_gamma,
            power=cfg.ema.power,
            model_cls=DiTTransformer2DModel,
            model_config=diffusion_model.config,
        )

    # Prepare everything with our `accelerator`.
    diffusion_model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        diffusion_model, optimizer, dataloader, lr_scheduler
    )

    # prepare decoder
    decoder.requires_grad_(False)
    decoder.eval()
    decoder.to(accelerator.device)

    if cfg.ema.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = (
        cfg.data_loader.params.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(len(dataloader) / accelerator.gradient_accumulation_steps)
    max_train_steps = cfg.num_epochs * num_update_steps_per_epoch

    num_diffusion_params = sum(p.numel() for p in diffusion_model.parameters())
    num_decoder_params = sum(p.numel() for p in decoder.parameters())

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_epochs}")
    logger.info(f"  Num Steps = {max_train_steps}")
    logger.info(f"  Batch size per device = {cfg.data_loader.params.batch_size}")
    logger.info(f"  Batch size total (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}\n")

    logger.info("***** #Parameters *****")
    logger.info(f"  Diffusion Model = {num_diffusion_params}")
    logger.info(f"  Decoder = {num_decoder_params}")
    logger.info(f"  Total = {num_diffusion_params+num_decoder_params}")

    global_step = 0
    first_epoch = 0

    # Train!
    for epoch in range(first_epoch, cfg.num_epochs):
        diffusion_model.train()  # enables dropout

        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            # Load latents and labels
            map_ids = batch["label"]
            latents = batch["input"]

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each latent
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()

            # Add noise to the clean latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            with accelerator.accumulate(diffusion_model):
                # Predict the noise residual
                model_output = diffusion_model(
                    hidden_states=noisy_latents,
                    class_labels=map_ids,
                    timestep=timesteps,
                ).sample
                loss = F.mse_loss(model_output, noise)  # TODO: add to config

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(diffusion_model.parameters(), 1.0)  # TODO: add to config
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.ema.use_ema:
                    ema_model.step(diffusion_model.parameters())
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if cfg.ema.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if cfg.debug_mode:
                break

        progress_bar.close()
        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % cfg.inference_epochs == 0 or epoch == cfg.num_epochs - 1:
                transformer = accelerator.unwrap_model(diffusion_model).eval()
                decoder = accelerator.unwrap_model(decoder)

                if cfg.ema.use_ema:
                    ema_model.store(transformer.parameters())
                    ema_model.copy_to(transformer.parameters())

                pipeline = LDMPipeline(
                    decoder=decoder,
                    transformer=transformer,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(cfg.seed)
                class_labels = list(range(cfg.num_classes)) * (cfg.inference_batch_size // cfg.num_classes)

                # (1) Create some synthetic images from the LDMPipeline
                sledge_vectors_generated = pipeline(
                    class_labels=class_labels,
                    num_inference_timesteps=cfg.num_inference_timesteps,
                    guidance_scale=cfg.guidance_scale,
                    generator=generator,
                    num_classes=cfg.num_classes,
                )
                generated_images = []
                for sledge_vector in sledge_vectors_generated:
                    generated_images.append(
                        get_sledge_vector_as_raster(sledge_vector.torch_to_numpy(), decoder._config)
                    )
                generated_images = np.array(generated_images).transpose(0, 3, 1, 2)

                # (2) Decode some latents from train set (for debugging)
                latents_norm = latents[: cfg.inference_batch_size]
                sledge_vectors_training = decoder.decode(latents_norm).unpack()
                training_images = []
                for sledge_vector in sledge_vectors_training:
                    training_images.append(get_sledge_vector_as_raster(sledge_vector.torch_to_numpy(), decoder._config))
                training_images = np.array(training_images).transpose(0, 3, 1, 2)

                # Add (1) + (2) to tensorboard
                tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                tracker.add_images("diff_generated", generated_images, epoch)
                tracker.add_images("diff_training", training_images, epoch)

                if cfg.ema.use_ema:
                    ema_model.restore(transformer.parameters())

            # save the model
            transformer = accelerator.unwrap_model(diffusion_model).eval()
            decoder = accelerator.unwrap_model(decoder)

            if cfg.ema.use_ema:
                ema_model.store(transformer.parameters())
                ema_model.copy_to(transformer.parameters())

            pipeline = LDMPipeline(
                decoder=decoder,
                transformer=transformer,
                scheduler=noise_scheduler,
            )
            pipeline.save_pretrained(Path(cfg.output_dir) / "checkpoint")

            if cfg.ema.use_ema:
                ema_model.restore(transformer.parameters())

    accelerator.end_training()
