# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 2 --device 2 

for dist in "normal" "laplace" "uniform" "binomial"
do
python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 2 \
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.012 --use_augmentation --nsm_distribution $dist
done