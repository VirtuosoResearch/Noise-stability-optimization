for seed in 0 1 2
do
python compute_hessian_traces.py --config configs/config_constraint_cifar10.json --use_augmentation  \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_Cifar10DataLoader_sam_0.01_ada_False_bs_256_unnormalize \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_cifar100.json --use_augmentation  \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_Cifar100DataLoader_sam_0.01_ada_False_bs_256_unnormalize \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_caltech.json --use_augmentation \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CaltechDataLoader_sam_0.01_ada_False_bs_32_unnormalize \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json --use_augmentation \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_sam_0.02_ada_False_bs_32_unnormalize \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_indoor.json --use_augmentation \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_IndoorDataLoader_sam_0.02_ada_False_bs_32_unnormalize \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_messidor.json --use_augmentation \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_MessidorDataLoader_sam_0.01_ada_False_bs_32_unnormalize \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed
done