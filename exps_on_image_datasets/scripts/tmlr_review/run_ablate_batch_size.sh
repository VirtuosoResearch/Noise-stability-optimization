# for bs in 8 16 32 64
# do
# epochs=$((30*$bs/32))
# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 2 --device 0 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.01 --batch_size $bs --epochs $epochs

# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 2 --device 0 --train_sam --sam_rho 0.02 --batch_size $bs --epochs $epochs
# done

for bs in 8 16 32 64
do
epochs=$((30*$bs/32))
python train_label_noise.py --config configs/config_constraint_aircrafts.json --use_augmentation\
    --model ResNet34 --lr 0.0001 --runs 2 --device 0 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.01 --batch_size $bs --epochs $epochs

python train_label_noise.py --config configs/config_constraint_aircrafts.json --use_augmentation\
    --model ResNet34 --lr 0.0001 --runs 2 --device 0 --train_sam --sam_rho 0.02 --batch_size $bs --epochs $epochs
done