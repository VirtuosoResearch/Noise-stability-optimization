python train_label_noise.py --config configs/config_constraint_cifar10.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_ls --ls_alpha 0.4

python train_label_noise.py --config configs/config_constraint_cifar100.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_ls --ls_alpha 0.4

python train_label_noise.py --config configs/config_constraint_cifar10.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 0\
    --train_swa --swa_epoch 20 --swa_lr 0.0002

python train_label_noise.py --config configs/config_constraint_cifar100.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 0\
    --train_swa --swa_epoch 20 --swa_lr 0.0002