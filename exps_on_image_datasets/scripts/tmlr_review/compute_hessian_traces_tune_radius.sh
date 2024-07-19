# for rho in 0.001 0.002 0.005 0.01 0.02 0.05
# do
# for seed in 0 1 2
# do
# python compute_hessian_traces.py --config configs/config_constraint_indoor.json --use_augmentation \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir "ResNet34_IndoorDataLoader_sam_${rho}_ada_False_bs_32" \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 2 --seed $seed
# done
# done

for rho in 0.001 0.002 0.005 0.01 0.02 0.05
do
for seed in 0 1 2
do
python compute_hessian_traces.py --config configs/config_constraint_indoor.json --use_augmentation \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir "ResNet34_IndoorDataLoader_sam_${rho}_ada_False_bs_32_unnormalize" \
    --checkpoint_name model_best \
    --sample_size 1000 --device 2 --seed $seed
done
done

# for bs in 8 16 32 64
# do
# for seed in 0 1 2
# do
# python compute_hessian_traces.py --config configs/config_constraint_indoor.json --use_augmentation \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir "ResNet34_IndoorDataLoader_nsm_0.0_0.01_1_True_distribution_normal_bs_${bs}" \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 2 --seed $seed
# done
# done


# for bs in 8 16 32 64
# do
# for seed in 0 1 2
# do
# python compute_hessian_traces.py --config configs/config_constraint_indoor.json --use_augmentation \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir "ResNet34_IndoorDataLoader_sam_0.02_ada_False_bs_${bs}" \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 2 --seed $seed
# done
# done


# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json --use_augmentation \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_sam_0.02_ada_False_bs_32_unnormalize \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed