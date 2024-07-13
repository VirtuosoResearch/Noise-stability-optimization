# for seed in 0 1 2
# do
# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_sam_0.1_ada_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed

# python compute_hessian_traces.py --config configs/config_constraint_birds.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_BirdsDataLoader_sam_0.1_ada_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed

# python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_IndoorDataLoader_sam_0.1_ada_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed

# python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_MessidorDataLoader_sam_0.1_ada_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed

# python compute_hessian_traces.py --config configs/config_constraint_aptos.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AptosDataLoader_sam_0.1_ada_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 1 --seed $seed
# done

for seed in 0 1 2
do
python compute_hessian_traces.py --config configs/config_constraint_cifar10.json  \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CIFAR10DataLoader_sam_0.1_ada_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 2 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_cifar100.json  \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CIFAR100DataLoader_sam_0.1_ada_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 2 --seed $seed
done