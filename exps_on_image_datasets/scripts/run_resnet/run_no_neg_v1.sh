python train_label_noise.py --config configs/config_constraint_cifar10.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 0 --train_nsm  --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.01

python train_label_noise.py --config configs/config_constraint_flower.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 0 --train_nsm  --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.01

python train_label_noise.py --config configs/config_constraint_cars.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 0 --train_nsm  --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.01
