for sigma in 0.01 0.02 0.03
do
python train_label_noise.py --config configs/config_constraint_indoor.json --use_augmentation\
    --model ResNet34 --lr 0.0001 --runs 2 --device 0 --train_nsm --use_neg --nsm_lam 1 --num_perturbs 3 \
    --nsm_sigma $sigma --nsm_sigma_schedule linear

python train_label_noise.py --config configs/config_constraint_indoor.json --use_augmentation\
    --model ResNet34 --lr 0.0001 --runs 2 --device 0 --train_nsm --use_neg --nsm_lam 1 --num_perturbs 3 \
    --nsm_sigma $sigma --nsm_sigma_schedule exp

python train_label_noise.py --config configs/config_constraint_aircrafts.json --use_augmentation\
    --model ResNet34 --lr 0.0001 --runs 2 --device 0 --train_nsm --use_neg --nsm_lam 1 --num_perturbs 3 \
    --nsm_sigma $sigma --nsm_sigma_schedule linear

python train_label_noise.py --config configs/config_constraint_aircrafts.json --use_augmentation\
    --model ResNet34 --lr 0.0001 --runs 2 --device 0 --train_nsm --use_neg --nsm_lam 1 --num_perturbs 3 \
    --nsm_sigma $sigma --nsm_sigma_schedule exp
done