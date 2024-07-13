# python train_label_noise.py --config configs/config_constraint_cifar100.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 1 --synthetic_noise --noise_rate 0.4

# python train_label_noise.py --config configs/config_constraint_cifar100.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 1 --synthetic_noise --noise_rate 0.6

python train_label_noise.py --config configs/config_constraint_cifar100.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 1 \
    --train_sam --sam_rho 0.1 --sam_adaptive

python train_label_noise.py --config configs/config_constraint_cifar100.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 1 \
    --synthetic_noise --noise_rate 0.4\
    --train_sam --sam_rho 0.1 --sam_adaptive