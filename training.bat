@echo off
set AZFUSE_USE_FUSE=0
set NCCL_ASYNC_ERROR_HANDLING=0
python finetune_sdm_yaml.py ^
    --cf config/disco_w_tm/yz_tiktok_S256L16_xformers_tsv_temdisco_temp_attn.py ^
    --do_train ^
    --root_dir D:\Yeshiva\3d_pose_generation ^
    --local_train_batch_size 2 ^
    --local_eval_batch_size 2 ^
    --log_dir exp/tiktok_ft2 ^
    --epochs 20 ^
    --deepspeed ^
    --eval_step 500 ^
    --save_step 500 ^
    --gradient_accumulate_steps 1 ^
    --learning_rate 1e-4 ^
    --fix_dist_seed ^
    --loss_target "noise" ^
    --train_yaml D:\Yeshiva\3d_pose_generation\dataset\TikTok_finetuning\composite_offset\train_TiktokDance-poses-masks.yaml ^
    --val_yaml D:\Yeshiva\3d_pose_generation\dataset\TikTok_finetuning\composite_offset/new10val_TiktokDance-poses-masks.yaml ^
    --unet_unfreeze_type "all" ^
    --refer_sdvae ^
    --ref_null_caption False ^
    --combine_clip_local ^
    --combine_use_mask ^
    --train_sample_interval 4 ^
    --nframe 16 ^
    --frame_interval 1 ^
    --conds "poses" "masks" ^
    --pretrained_model /media/namrata/Projects/Yeshiva/3d_pose_generation/exp/tiktok_ft/mp_rank_00_model_states.pt
pause
