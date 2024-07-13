for seed in 0 1 2
do
python compute_hessian_traces.py --config configs/config_constraint_cifar10.json  \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_Cifar10DataLoader_bsam_0.01_0.01 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_cifar100.json  \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_Cifar100DataLoader_bsam_0.01_0.01 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_caltech.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CaltechDataLoader_bsam_0.01_0.01 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_birds.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_BirdsDataLoader_bsam_0.01_0.01 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --seed $seed
done