import torch

from config import *
from config.ref_attn_clip_combine_controlnet.net import Net, inner_collect_fn

class Args(BasicArgs):
    task_name, method_name = BasicArgs.parse_config_name(__file__)
    log_dir = os.path.join(BasicArgs.root_dir, task_name, method_name)

    # data
    dataset_cf = 'dataset/tiktok_controlnet_t2i_imagevar_combine_tsv.py'
    max_train_samples = None
    max_eval_samples = None
    # max_eval_samples = 2
    max_video_len = 1  # L=16
    debug_max_video_len = 1
    img_full_size = (256, 256)
    img_size = (256, 256)
    fps = 5
    data_dir = "/scratch/namrata/3d_pose_gen/dataset"
    # debug_train_yaml = 'Titktok_finetuning/composite_offset/train_TiktokDance-poses-masks.yaml'
    # debug_val_yaml = 'Titktok_finetuning/composite_offset/new10val_TiktokDance-poses-masks.yaml'
    # train_yaml = 'Titktok_finetuning/composite_offset/train_TiktokDance-poses-masks.yaml'
    # val_yaml = 'Titktok_finetuning/composite_offset/new10val_TiktokDance-poses-masks.yaml'
    debug_train_yaml=""
    debug_val_yaml=""
    train_yaml=""
    val_yaml=""

    # WT: for tiktok image data dir
    tiktok_data_root = 'dataset/Tiktok_finetuning/TiktokDance'
    # tiktok_ann_root = 'keli/dataset/TikTok_dataset/pair_ann'
    refer_sdvae = False

    eval_before_train = True
    eval_enc_dec_only = False

    # training
    local_train_batch_size = 8
    local_eval_batch_size = 8
    learning_rate = 3e-5
    null_caption = False
    refer_sdvae = False


    # max_norm = 1.0
    epochs = 50
    num_workers = 4
    eval_step = 5
    save_step = 5
    drop_text = 1.0 # drop text only activate in args.null_caption, default=1.0
    scale_factor = 0.18215
    # pretrained_model_path = os.path.join(BasicArgs.root_dir, 'diffusers/stable-diffusion-v2-1')
    pretrained_model_path = os.path.join(BasicArgs.root_dir, 'diffusers/sd-image-variations-diffusers')
    sd15_path = os.path.join(BasicArgs.root_dir, 'diffusers/stable-diffusion-v1-5-2')
    gradient_checkpointing = True
    enable_xformers_memory_efficient_attention = True
    freeze_unet=True

    # sample
    num_inf_images_per_prompt = 1
    num_inference_steps = 50
    guidance_scale = 7.5

    # others
    seed = 42
    # set_seed(seed)

args = Args()
