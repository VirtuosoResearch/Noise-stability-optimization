for sigma in 0.01 0.005 0.015
do
for perturb in 5 10
do
python train_label_noise.py --config configs/config_constraint_cifar100.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 1 --synthetic_noise --noise_rate 0.4\
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs $perturb --nsm_sigma $sigma
done
done
# 0.02 0.05 0.1
