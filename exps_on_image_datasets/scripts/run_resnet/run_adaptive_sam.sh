# for rho in 0.1 0.5 1 2
# do
# python train_label_noise.py --config configs/config_constraint_indoor.json \
#     --model ResNet34 --lr 0.0001 --runs 1 --device 1 --train_sam --sam_rho $rho  --sam_adaptive
# done

python train_label_noise.py --config configs/config_constraint_cifar10.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 0\
    --train_sam --sam_rho 0.1 --sam_adaptive

python train_label_noise.py --config configs/config_constraint_cifar100.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 0 \
    --train_sam --sam_rho 0.1 --sam_adaptive

for seed in 0 1 2
do
python compute_hessian_traces.py --config configs/config_constraint_cifar10.json  \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_Cifar10DataLoader_sam_0.1_ada_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_cifar100.json  \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_Cifar100DataLoader_sam_0.1_ada_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --seed $seed
done