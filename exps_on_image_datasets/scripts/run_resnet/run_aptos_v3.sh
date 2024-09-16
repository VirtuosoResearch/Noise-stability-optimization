for sigma in 0.008 0.01 0.012
do
python train_label_noise.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --lr 0.0001 --runs 3 --device 3 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 3 --nsm_sigma $sigma
done