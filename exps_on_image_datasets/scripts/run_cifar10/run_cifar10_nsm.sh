for sigma in 0.01 0.008 0.012
do
python train_label_noise.py --config configs/config_constraint_cifar10.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 2 --synthetic_noise --noise_rate 0.4\
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma $sigma
done
# 0.02 0.05 0.1
