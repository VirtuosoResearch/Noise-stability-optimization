# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 2 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.01

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 2 --train_sam --sam_rho 0.01

# python compute_hessian_traces_v2.py --config configs/config_constraint_cifar10.json \
#     --model ResNet34 --batch_size 4 \
#     --checkpoint_dir ResNet34_Cifar10DataLoader_nsm_0.0_0.01_1_True \
#     --checkpoint_name model_best \
#     --sample_size 10 --device 2

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 1

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 1 --train_sam --sam_rho 0.01

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.01

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.012

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 0 \
#     --train_nsmswa --swa_epoch 20 --swa_lr 0.0002 --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.008

# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.008

python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 0 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.008 \
    --epochs 15

python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 0 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 3 --nsm_sigma 0.008 \
    --epochs 10

python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 1 --train_nsm --nsm_lam 1 --num_perturbs 4 --nsm_sigma 0.01 \
    --epochs 15

python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 1 --train_nsm --nsm_lam 1 --num_perturbs 6 --nsm_sigma 0.01 \
    --epochs 10
