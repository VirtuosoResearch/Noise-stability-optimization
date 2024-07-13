for wd in 1e-4 1e-3
do
# python train_label_noise.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --lr 0.0001 --runs 2 --device 1 \
#     --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.008 --weight_decay $wd

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 1 \
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.012 --weight_decay $wd --use_augmentation
done