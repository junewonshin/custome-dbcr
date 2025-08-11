"""
Train a diffusion model on images.
"""

import argparse

from ddbm_fixed import dist_util, logger
from datasets import load_data
from ddbm_fixed.resample import create_named_schedule_sampler
from ddbm_fixed.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
    get_workdir,
)
from ddbm_fixed.random_util import seed_all
from ddbm_fixed.train_util import TrainLoop

import torch.distributed as dist
from pathlib import Path

from glob import glob
import os
from datasets.augment import AugmentPipe


def main(args):
    seed_all(42)

    workdir = get_workdir(args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=workdir)
    if dist.get_rank() == 0:
        logger.log("creating model and diffusion...")

    data_image_size = args.image_size
    # Load target model
    resume_train_flag = False
    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f"{workdir}/*model*[0-9].*"))
        if len(model_ckpts) > 0:
            max_ckpt = max(model_ckpts, key=lambda x: int(x.split("model_")[-1].split(".")[0]))
            if os.path.exists(max_ckpt):
                args.resume_checkpoint = max_ckpt
                resume_train_flag = True
        elif args.pretrained_ckpt is not None:
            max_ckpt = args.pretrained_ckpt
            args.resume_checkpoint = max_ckpt
        if dist.get_rank() == 0 and args.resume_checkpoint != "":
            logger.log("Resuming from checkpoint: ", max_ckpt)

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))  
    # print(f"[RANK {dist.get_rank()}] named param list: {[name for name, _ in model.named_parameters()]}")
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}")
    else:
        batch_size = args.batch_size

    if dist.get_rank() == 0:
        logger.log("creating data loader...")

    data, test_data = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=batch_size,
        image_size=data_image_size,
        num_workers=args.num_workers,
    )

    if args.use_augment:
        augment = AugmentPipe(p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
    else:
        augment = None

    if dist.get_rank() == 0:
        logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_data=data,
        test_data=test_data,
        batch_size=batch_size,
        microbatch=-1 if args.microbatch >= batch_size else args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval_epochs=args.log_interval_epochs,
        test_interval_epochs=args.test_interval_epochs,
        save_interval_epochs=args.save_interval_epochs,
        save_interval_for_preemption_epochs=args.save_interval_for_preemption_epochs,
        resume_checkpoint=args.resume_checkpoint,
        workdir=workdir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        total_epochs=args.total_epochs,
        augment_pipe=augment,
        train_mode=args.train_mode,
        resume_train_flag=resume_train_flag,
        **sample_defaults(),
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/home/work/dataset/SEN12MSCR",
        dataset="sen12mscr",
        schedule_sampler="real-uniform",
        lr=2e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=32,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval_epochs=1,
        test_interval_epochs=1,
        save_interval_epochs=5,
        save_interval_for_preemption_epochs=10,
        resume_checkpoint="",
        exp="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=16,
        use_augment=False,
        pretrained_ckpt=None,
        train_mode="ddbm",
        total_epochs=200,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)


