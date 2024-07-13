python train_label_noise.py --config configs/config_constraint_cifar10.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 2 --synthetic_noise --noise_rate 0.4\
    --train_swa --swa_epoch 20 --swa_lr 0.0005