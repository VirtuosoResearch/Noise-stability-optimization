for rho in 0.02 0.05 0.1
do
python train_label_noise.py --config configs/config_constraint_cifar10.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 1 --synthetic_noise --noise_rate 0.4\
    --train_sam --sam_rho $rho
done

