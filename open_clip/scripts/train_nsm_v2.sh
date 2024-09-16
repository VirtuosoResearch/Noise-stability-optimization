# CUDA_VISIBLE_DEVICES=0 python src/open_clip_train/main.py \
#     --save-frequency 1 \
#     --zeroshot-frequency 1 \
#     --report-to tensorboard \
#     --train-data ./cc3m_train/{00000..00126}.tar  \
#     --val-data ./cc3m_validation/{00000..00001}.tar   \
#     --train-num-samples 1000000\
#     --val-num-samples 15840\
#     --warmup 10000 \
#     --batch-size 128 \
#     --lr 1e-3 \
#     --wd 0.1 \
#     --epochs 30 \
#     --workers 8 \
#     --model ViT-S-16\
#     --train_nsm --nsm_use_neg --nsm_lam 0 --nsm_num_perturbs 1 --nsm_sigma 0.01 --name train_nsm_sigma_0.01_new\
#     --resume ./logs/train_nsm_sigma_0.01/checkpoints/epoch_1.pt

CUDA_VISIBLE_DEVICES=1 python src/open_clip_train/main.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data ./cc3m_train/{00000..00126}.tar  \
    --val-data ./cc3m_validation/{00000..00001}.tar   \
    --train-num-samples 1000000\
    --val-num-samples 15840\
    --warmup 10000 \
    --batch-size 128 \
    --lr 1e-3 \
    --wd 0.1 \
    --epochs 30 \
    --workers 8 \
    --model ViT-S-16\
    --train_nsm --nsm_use_neg --nsm_lam 0 --nsm_num_perturbs 1 --nsm_sigma 0.015 --name train_nsm_sigma_0.015