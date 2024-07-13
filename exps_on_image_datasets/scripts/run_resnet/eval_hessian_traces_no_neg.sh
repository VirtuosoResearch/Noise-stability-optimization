python compute_hessian_traces.py --config configs/config_constraint_cifar10.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_Cifar10DataLoader_nsm_0.0_0.01_2_False \
    --checkpoint_name model_epoch_10 \
    --sample_size 1000 --device 0

python compute_hessian_traces.py --config configs/config_constraint_cifar100.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_Cifar100DataLoader_nsm_0.0_0.01_2_False \
    --checkpoint_name model_epoch_10 \
    --sample_size 1000 --device 0

python compute_hessian_traces.py --config configs/config_constraint_flower.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_FlowerDataLoader_nsm_0.0_0.01_2_False \
    --checkpoint_name model_epoch_10 \
    --sample_size 1000 --device 0

python compute_hessian_traces.py --config configs/config_constraint_cars.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CarsDataLoader_nsm_0.0_0.01_2_False \
    --checkpoint_name model_epoch_10 \
    --sample_size 1000 --device 0

python compute_hessian_traces.py --config configs/config_constraint_birds.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_BirdsDataLoader_nsm_0.0_0.01_2_False \
    --checkpoint_name model_epoch_10 \
    --sample_size 1000 --device 0


python compute_hessian_traces.py --config configs/config_constraint_caltech.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CaltechDataLoader_nsm_0.0_0.01_2_False \
    --checkpoint_name model_epoch_10 \
    --sample_size 1000 --device 0