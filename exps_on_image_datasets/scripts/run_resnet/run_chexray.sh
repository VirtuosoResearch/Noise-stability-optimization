python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 3 --device 1 --data_frac 0.1 --downsample_test


python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 2 --device 1 --data_frac 0.1 --downsample_test\
    --train_sam --sam_rho 0.01


python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 2 --device 1 --data_frac 0.1 --downsample_test\
    --train_sam --sam_rho 0.1 --sam_adaptive

python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 1 --device 1 --data_frac 0.1 --downsample_test\
    --train_sam --sam_rho 0.02

python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 1 --device 1 --data_frac 0.1 --downsample_test\
    --train_sam --sam_rho 0.2 --sam_adaptive

python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 1 --device 1 --data_frac 0.1 --downsample_test\
    --train_sam --sam_rho 0.05

python train_label_noise.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --lr 0.0005 --runs 1 --device 1 --data_frac 0.1 --downsample_test\
    --train_sam --sam_rho 0.5 --sam_adaptive