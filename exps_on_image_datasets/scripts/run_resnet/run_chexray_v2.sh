python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 1 --device 0 --data_frac 0.1 --downsample_test\
    --train_nsm --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.01

python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 1 --device 0 --data_frac 0.1 --downsample_test\
    --train_nsm --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.012

python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 1 --device 0 --data_frac 0.1 --downsample_test\
    --train_nsm --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.008

python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 1 --device 0 --data_frac 0.1 --downsample_test\
    --train_nsm --train_nsm --use_neg --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.01

python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 1 --device 0 --data_frac 0.1 --downsample_test\
    --train_nsm --train_nsm --use_neg --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.012

python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 1 --device 0 --data_frac 0.1 --downsample_test\
    --train_nsm --train_nsm --use_neg --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.008