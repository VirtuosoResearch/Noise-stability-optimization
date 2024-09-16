for seed in 1 2 3
do
python compute_hessian_traces.py --config configs/config_constraint_cifar10.json \
    --model ResNet34 --batch_size 4 \
    --checkpoint_dir ResNet34_Cifar10DataLoader_ls_0.4 \
    --checkpoint_name model_best \
    --sample_size 500 --device 1 --seed $seed
done

for seed in 1 2 3
do
python compute_hessian_traces.py --config configs/config_constraint_cifar100.json \
    --model ResNet34 --batch_size 4 \
    --checkpoint_dir ResNet34_Cifar100DataLoader_ls_0.4 \
    --checkpoint_name model_best \
    --sample_size 500 --device 1 --seed $seed
done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar10.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_Cifar10DataLoader_nsm_0.0_0.01_4_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar100.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_Cifar100DataLoader_nsm_0.0_0.01_4_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done

# # CIFAR 10

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar10.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_Cifar10DataLoader_none_none_1.0000_1.0000_run_0 \
#     --checkpoint_name model_epoch_30 \
#     --sample_size 1000 --device 1 --seed $seed
# done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar10.json \
#     --model ResNet34 --batch_size 4 \
#     --checkpoint_dir ResNet34_Cifar10DataLoader_sam_0.01 \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar10.json \
#     --model ResNet34 --batch_size 4 \
#     --checkpoint_dir ResNet34_Cifar10DataLoader_nsm_0.0_0.01_1_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done

# # CIFAR 100

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar100.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_Cifar100DataLoader_none_none_1.0000_1.0000_run_0 \
#     --checkpoint_name model_epoch_30 \
#     --sample_size 1000 --device 1 --seed $seed
# done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar100.json \
#     --model ResNet34 --batch_size 4 \
#     --checkpoint_dir ResNet34_Cifar100DataLoader_sam_0.01 \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar100.json \
#     --model ResNet34 --batch_size 4 \
#     --checkpoint_dir ResNet34_Cifar100DataLoader_nsm_0.0_0.01_1_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done

# # More peturbations

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar10.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_Cifar10DataLoader_nsm_0.0_0.01_2_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar10.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_Cifar10DataLoader_nsm_0.0_0.01_3_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar100.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_Cifar100DataLoader_nsm_0.0_0.01_2_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_cifar100.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_Cifar100DataLoader_nsm_0.0_0.01_3_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done