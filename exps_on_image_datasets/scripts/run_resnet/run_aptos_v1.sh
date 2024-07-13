python train_label_noise.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_sam --sam_rho 0.01


for alpha in 0.2 0.4 
do
python train_label_noise.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_ls --ls_alpha $alpha
done

python train_label_noise.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_sam --sam_rho 0.02

python train_label_noise.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 0 --train_sam --sam_rho 0.05