for sigma in 0.01 0.008 0.012
do
python train_label_noise.py --config configs/config_constraint_messidor.json \
    --model ResNet34 \
    --lr 0.0001 --runs 3 --device 3 \
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma $sigma
done