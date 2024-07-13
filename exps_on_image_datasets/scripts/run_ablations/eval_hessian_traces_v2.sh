for seed in 0
do
# for i in 5 10 15 20 25 30
# do
# python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_IndoorDataLoader_nsm_0.0_0.012_2_True \
#     --checkpoint_name model_epoch_${i} \
#     --sample_size 1000 --device 2 --seed $seed
# done

# for i in 5 10 15 20 25 30
# do
# python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_IndoorDataLoader_nsm_0.0_0.012_2_True_aug \
#     --checkpoint_name model_epoch_${i} \
#     --sample_size 1000 --device 2 --seed $seed
# done

# for i in 5 10 15 20 25 30
# do
# python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_IndoorDataLoader_nsm_0.0_0.012_2_True_wd_0.001 \
#     --checkpoint_name model_epoch_${i} \
#     --sample_size 1000 --device 2 --seed $seed
# done

# for i in 5 10 15 20 25 30
# do
# python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_IndoorDataLoader_nsm_0.0_0.012_2_True_wd_0.0001 \
#     --checkpoint_name model_epoch_${i} \
#     --sample_size 1000 --device 2 --seed $seed
# done

# for i in 5 10 15 20 25 30
# do
# python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_IndoorDataLoader_nsm_0.0_0.012_1_True_distribution_normal_aug_penalty \
#     --checkpoint_name model_epoch_${i} \
#     --sample_size 1000 --device 2 --seed $seed --use_augmentation
# done

# for i in 5 10 15 20 25 30
# do
# python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_IndoorDataLoader_nsm_0.0_0.012_1_True_distribution_normal_wd_0.0001_aug \
#     --checkpoint_name model_epoch_${i} \
#     --sample_size 1000 --device 2 --seed $seed --use_augmentation
# done

python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_IndoorDataLoader_nsm_0.0_0.012_1_False_distribution_normal_aug \
    --checkpoint_name model_best \
    --sample_size 1000 --device 2 --seed $seed --use_augmentation

python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_IndoorDataLoader_nsm_0.0_0.012_1_True_distribution_laplace_aug \
    --checkpoint_name model_best \
    --sample_size 1000 --device 2 --seed $seed --use_augmentation

python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_IndoorDataLoader_nsm_0.0_0.012_1_True_distribution_cauchy_aug \
    --checkpoint_name model_best \
    --sample_size 1000 --device 2 --seed $seed --use_augmentation
done