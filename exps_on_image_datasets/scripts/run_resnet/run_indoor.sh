# for alpha in 0.2 0.4 
# do
# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 3 --train_ls --ls_alpha $alpha
# done

# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 3 \
#     --train_swa --swa_epoch 20 --swa_lr 0.0002

# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 3 \
#     --train_swa --swa_epoch 20 --swa_lr 0.0001

# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 3 \
#     --train_nsmswa --swa_epoch 20 --swa_lr 0.0002 --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.012

# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 3 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.012

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 3 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.012 \
    --epochs 15

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 3 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 3 --nsm_sigma 0.012 \
    --epochs 10

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --nsm_lam 0 --num_perturbs 4 --nsm_sigma 0.01 \
    --epochs 15 

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --nsm_lam 0 --num_perturbs 6 --nsm_sigma 0.01 \
    --epochs 10
