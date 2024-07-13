for wd in 1e-4
do
python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --lr 0.0001 --runs 2 --device 1 \
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.012 --use_augmentation \
    --reg_method penalty --reg_extractor $wd --reg_predictor $wd --scale_factor 1
done