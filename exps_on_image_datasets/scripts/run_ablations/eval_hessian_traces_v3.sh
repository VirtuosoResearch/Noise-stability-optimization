for seed in 0
do

for i in 5 10 15 20 25 30
do
python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_IndoorDataLoader_nsm_0.0_0.012_1_True_distribution_normal_aug_penalty \
    --checkpoint_name model_epoch_${i} \
    --sample_size 1000 --device 3 --seed $seed --use_augmentation
done

done