for rho in 0.01
do
python train_label_noise.py --config configs/config_constraint_cifar100.json  --use_augmentation\
    --model ResNet34 --lr 0.0001 --runs 2 --device 2 --train_sam --sam_rho $rho --sam_unnormalize

python train_label_noise.py --config configs/config_constraint_caltech.json  --use_augmentation\
    --model ResNet34 --lr 0.0001 --runs 2 --device 2 --train_sam --sam_rho $rho --sam_unnormalize

python train_label_noise.py --config configs/config_constraint_messidor.json  --use_augmentation\
    --model ResNet34 --lr 0.0001 --runs 2 --device 2 --train_sam --sam_rho $rho --sam_unnormalize
done
