
# for i in 5 10 15 20 25 30
# do
# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_0.0_0.008_1_True \
#     --checkpoint_name model_epoch_${i} \
#     --sample_size 1000 --device 3 --seed $seed
# done

# for i in 5 10 15 20 25 30
# do
# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_0.0_0.008_1_True_aug \
#     --checkpoint_name model_epoch_${i} \
#     --sample_size 1000 --device 3 --seed $seed
# done

# for i in 5 10 15 20 25 30
# do
# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_0.0_0.008_1_True_wd_0.001 \
#     --checkpoint_name model_epoch_${i} \
#     --sample_size 1000 --device 3 --seed $seed
# done

# for i in 5 10 15 20 25 30
# do
# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_0.0_0.008_1_True_wd_0.0001 \
#     --checkpoint_name model_epoch_${i} \
#     --sample_size 1000 --device 3 --seed $seed
# done

for seed in 0
do
python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_0.0_0.008_1_False_distribution_normal_aug \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --seed $seed --use_augmentation

python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_0.0_0.008_1_True_distribution_laplace_aug \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --seed $seed --use_augmentation

python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_0.0_0.008_1_True_distribution_cauchy_aug \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --seed $seed --use_augmentation
done