python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 3 \
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.008

python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 3 \
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.008 --use_augmentation

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 3 \
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.012

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 3 \
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.012 --use_augmentation