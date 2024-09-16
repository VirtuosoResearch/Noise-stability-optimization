# python train_label_noise.py --config configs/config_constraint_birds.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 2

# python train_label_noise.py --config configs/config_constraint_birds.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 2 --train_sam --sam_rho 0.01

python train_label_noise.py --config configs/config_constraint_birds.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.012

python train_label_noise.py --config configs/config_constraint_birds.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.012

python train_label_noise.py --config configs/config_constraint_birds.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 3 --nsm_sigma 0.012
