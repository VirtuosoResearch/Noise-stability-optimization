# python train_label_noise.py --config configs/config_constraint_aptos.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 3 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.02

# python train_label_noise.py --config configs/config_constraint_aptos.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 3 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.03

# python train_label_noise.py --config configs/config_constraint_aptos.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 3 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.05

for sigma in 0.008 0.01 0.012
do
python train_label_noise.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 2 --nsm_sigma $sigma
done
