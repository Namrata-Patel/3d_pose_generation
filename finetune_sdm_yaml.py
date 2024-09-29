# --------------------------------------------------------
# DisCo - Disentangled Control for Referring Human Dance Generation in Real World
# Licensed under The Apache-2.0 license License [see LICENSE for details]
# Tan Wang (TAN317@e.ntu.edu.sg)
# Work done during internship at Microsoft
# --------------------------------------------------------

from utils.wutils_ldm import *
from agent import Agent_LDM, WarmupLinearLR, WarmupLinearConstantLR
import os
import torch
from utils.lib import *
from utils.dist import dist_init
from dataset.tsv_dataset import make_data_sampler, make_batch_data_sampler
torch.multiprocessing.set_sharing_strategy('file_system')
from deepspeed.ops.adam import DeepSpeedCPUAdam

def get_loader_info(args, size_batch, dataset):
    # Determine if the dataset is for training
    is_train = dataset.split == 'train'

    if is_train:
        # Calculate the number of images each GPU will handle,
        # capped at a maximum of 128
        images_per_gpu = min(
            size_batch * max(1, (args.max_video_len // dataset.max_video_len)),
            128)
        
        # Calculate the total number of images processed per batch
        images_per_batch = images_per_gpu * args.world_size
        
        # Calculate the number of iterations required per epoch
        iter_per_ep = len(dataset) // images_per_batch

        # If epochs are not specified, calculate epochs from iterations
        if args.epochs == -1:
            assert args.ft_iters > 0  # Ensure that fine-tuning iterations are positive
            num_iters = args.ft_iters
            args.epochs = (num_iters * images_per_batch) // len(dataset) + 1
        else:
            # Calculate the total number of iterations based on epochs
            num_iters = iter_per_ep * args.epochs
    else:
        # For non-training datasets, calculate the number of images per GPU
        images_per_gpu = size_batch * (
            args.max_video_len // dataset.max_video_len)
        
        # Calculate the total number of images processed per batch
        images_per_batch = images_per_gpu * args.world_size
        
        # No need to calculate iterations per epoch or total iterations
        iter_per_ep = None
        num_iters = None

    # Return the calculated loader information
    loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
    return loader_info



def make_data_loader(
        args, size_batch, dataset, start_iter=0, loader_info=None):
    # Determine if the dataset is for training
    is_train = dataset.split == 'train'
    
    # Set the collate function (currently not used)
    collate_fn = None #dataset.collate_batch
    
    # Check if distributed training is enabled
    is_distributed = args.distributed
    
    # Set shuffle and start_iter based on whether it's training or not
    if is_train:
        shuffle = True
        start_iter = start_iter
    else:
        shuffle = False
        start_iter = 0
    
    # Get loader information if not provided
    if loader_info is None:
        loader_info = get_loader_info(args, size_batch, dataset)
    images_per_gpu, images_per_batch, iter_per_ep, num_iters = loader_info

    # Determine the number of limited samples if specified
    if hasattr(args, 'limited_samples'):
        limited_samples = args.limited_samples // args.local_size
    else:
        limited_samples = -1
    
    # Set random seed for reproducibility
    random_seed = args.seed
    
    # Create a data sampler with the specified settings
    sampler = make_data_sampler(
        dataset, shuffle, is_distributed, limited_samples=limited_samples,
        random_seed=random_seed)
    
    # Create a batch sampler using the data sampler
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    
    # Create the DataLoader with the batch sampler
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True, collate_fn=collate_fn
    )
    
    # Prepare metadata about the data loading process
    meta_info = (images_per_batch, iter_per_ep, num_iters)
    
    return data_loader, meta_info


def main_worker(args):
    """
    Main function to handle the training and evaluation process based on the provided arguments.
    """

    # Import configuration files and classes
    cf = import_filename(args.cf)
    Net, inner_collect_fn = cf.Net, cf.inner_collect_fn

    dataset_cf = import_filename(args.dataset_cf)
    BaseDataset = dataset_cf.BaseDataset

    # Print sorted arguments for debugging purposes
    print(f"Args: {edict(sorted(vars(args).items()))}")

    # Initialize models
    logger.info('Building models...')
    model = Net(args)  # Create an instance of the model
    
    # Check if training is to be performed
    if args.do_train:
        logger.warning("Do training...")

        # Prepare the training and evaluation datasets
        if getattr(args, 'refer_clip_preprocess', None):
            # Use the feature extractor if specified
            train_dataset = BaseDataset(args, args.train_yaml, split='train', preprocesser=model.feature_extractor)
            eval_dataset = BaseDataset(args, args.val_yaml, split='val', preprocesser=model.feature_extractor)
        else:
            train_dataset = BaseDataset(args, args.train_yaml, split='train')
            eval_dataset = BaseDataset(args, args.val_yaml, split='val')

        # Get loader information for training
        train_info = get_loader_info(args, args.local_train_batch_size, train_dataset)
        _, images_per_batch, args.iter_per_ep, args.num_iters = train_info

        # Adjust evaluation and checkpoint save steps
        if args.eval_step <= 5.0:
            args.eval_step = args.eval_step * args.iter_per_ep
        if args.save_step <= 5.0:
            args.save_step = args.save_step * args.iter_per_ep
        
        args.eval_step = int(max(10, args.eval_step))
        args.save_step = int(max(10, args.save_step))

        # Initialize optimizer
        from torch.optim import AdamW 
        # Use DeepSpeedCPUAdam if specified; otherwise, fallback to default optimizer
        optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-3)
        optimizer = getattr(model, 'optimizer', optimizer)

        # Initialize learning rate scheduler
        if args.constant_lr:
            scheduler = WarmupLinearConstantLR(
                optimizer,
                max_iter=(args.num_iters // args.gradient_accumulate_steps) + 1,
                warmup_ratio=getattr(args, 'warmup_ratio', 0.05))
        else:
            scheduler = WarmupLinearLR(
                optimizer,
                max_iter=(args.num_iters // args.gradient_accumulate_steps) + 1,
                warmup_ratio=getattr(args, 'warmup_ratio', 0.05))
        scheduler = getattr(model, 'scheduler', scheduler)

        # Set up the trainer with model, optimizer, and scheduler
        trainer = Agent_LDM(args, model, optimizer, scheduler)
        trainer.setup_model_for_training()
        
        # Create data loaders for training and evaluation
        train_dataloader, train_info = make_data_loader(
            args, args.local_train_batch_size, 
            train_dataset, start_iter=trainer.global_step+1, loader_info=train_info)
        eval_dataloader, eval_info = make_data_loader(
            args, args.local_eval_batch_size, 
            eval_dataset)

        # Log training details
        logger.info(f"Video Length {train_dataset.size_frame}")
        logger.info(f"Total batch size {images_per_batch}")
        logger.info(f"Total training steps {args.num_iters}")
        logger.info(f"Starting train iter: {trainer.global_step+1}")
        logger.info(f"Training steps per epoch (accumulated) {args.iter_per_ep}")
        logger.info(f"Training dataloader length {len(train_dataloader)}")
        logger.info(f"Evaluation happens every {args.eval_step} steps")
        logger.info(f"Checkpoint saves every {args.save_step} steps")

        # Start training and evaluation
        trainer.train_eval_by_iter(
            train_loader=train_dataloader, 
            eval_loader=eval_dataloader,  
            inner_collect_fn=inner_collect_fn
        )

    # Check if evaluation visualization is needed
    if args.eval_visu:
        logger.warning("Do eval_visu...")
        
        # Prepare evaluation dataset
        if getattr(args, 'refer_clip_preprocess', None):
            eval_dataset = BaseDataset(args, args.val_yaml, split='val', preprocesser=model.feature_extractor)
        else:
            eval_dataset = BaseDataset(args, args.val_yaml, split='val')
        
        # Create data loader for evaluation
        eval_dataloader, eval_info = make_data_loader(
            args, args.local_eval_batch_size, 
            eval_dataset
        )

        # Initialize and run the evaluation
        trainer = Agent_LDM(args=args, model=model)
        trainer.eval(
            eval_dataloader, 
            inner_collect_fn=inner_collect_fn,
            enc_dec_only='enc_dec_only' in args.eval_save_filename
        )


if __name__ == "__main__":
    from utils.args import sharedArgs
    parsed_args = sharedArgs.parse_args()
    main_worker(parsed_args)
