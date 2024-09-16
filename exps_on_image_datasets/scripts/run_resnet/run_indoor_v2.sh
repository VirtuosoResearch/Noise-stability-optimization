python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.01

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --nsm_lam 0 --num_perturbs 4 --nsm_sigma 0.01

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --nsm_lam 0 --num_perturbs 6 --nsm_sigma 0.01
