# for alpha in 0.2 0.4 
# do
# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 1 --train_ls --ls_alpha $alpha
# done

python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 1 \
    --train_swa --swa_epoch 20 --swa_lr 0.0002

python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 1 \
    --train_swa --swa_epoch 20 --swa_lr 0.0001

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 1 --train_sam --sam_rho 0.01

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.01

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.012

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.008

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 1 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.01

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 1 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 3 --nsm_sigma 0.01