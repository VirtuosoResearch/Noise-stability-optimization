# for alpha in 0.2 0.4 
# do
# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 3 --device 3 --train_ls --ls_alpha $alpha
# done

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 3 \
    --train_swa --swa_epoch 20 --swa_lr 0.0002

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 3 \
    --train_swa --swa_epoch 20 --swa_lr 0.0001