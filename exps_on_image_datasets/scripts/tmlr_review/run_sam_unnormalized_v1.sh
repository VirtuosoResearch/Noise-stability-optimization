# for rho in 0.001 0.002 0.005 0.01 0.02 0.05 
# do
# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 2 --device 1 --train_sam --sam_rho $rho --sam_unnormalize
# done

for rho in 0.001 0.002 0.005 0.01 0.02 0.05 
do
python train_label_noise.py --config configs/config_constraint_aircrafts.json --use_augmentation\
    --model ResNet34 --lr 0.0001 --runs 2 --device 1 --train_sam --sam_rho $rho --sam_unnormalize
done

# for rho in 0.01
# do
# python train_label_noise.py --config configs/config_constraint_cifar10.json \
#     --model ResNet34 --lr 0.0001 --runs 2 --device 1 --train_sam --sam_rho $rho --sam_unnormalize
# done
