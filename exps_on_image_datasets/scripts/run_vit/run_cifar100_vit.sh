python train_label_noise.py --config configs/config_constraint_cifar100.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --lr 0.0001 --runs 1 --device 2 --batch_size 64 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.008

python train_label_noise.py --config configs/config_constraint_cifar100.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --lr 0.0001 --runs 1 --device 2 --batch_size 64 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.01

python train_label_noise.py --config configs/config_constraint_cifar100.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --lr 0.0001 --runs 1 --device 2 --batch_size 64 --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.005