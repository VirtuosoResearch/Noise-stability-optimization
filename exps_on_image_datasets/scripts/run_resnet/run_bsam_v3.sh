for rho in 0.01 0.02 0.05 
do
for sigma in 0.01 0.012 0.014
do
python train_label_noise.py --config configs/config_constraint_birds.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 1 --train_bsam --bsam_rho $rho --bsam_sigma $sigma
done
done

for rho in 0.01 0.02 0.05 
do
for sigma in 0.01 0.012 0.014
do
python train_label_noise.py --config configs/config_constraint_caltech.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 1 --train_bsam --bsam_rho $rho --bsam_sigma $sigma
done
done
