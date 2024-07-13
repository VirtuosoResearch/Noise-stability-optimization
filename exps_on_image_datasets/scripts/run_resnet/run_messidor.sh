python train_label_noise.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_sam --sam_rho 0.01

for alpha in 0.2 0.4 
do
python train_label_noise.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_ls --ls_alpha $alpha
done

python train_label_noise.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.01

python train_label_noise.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.012

python train_label_noise.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 2 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.014

